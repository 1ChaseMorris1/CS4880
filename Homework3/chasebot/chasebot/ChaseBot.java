package ai.chasebot;

import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import ai.chasebot.strategiesV3.AMilDefense;
import ai.chasebot.strategiesV3.AMilRush;
import ai.chasebot.strategiesV3.AWorkDefense;
import ai.chasebot.strategiesV3.AWorkRush;
import ai.chasebot.strategiesV3.AwareAI;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitTypeTable;

import java.util.ArrayList;
import java.util.List;

public class ChaseBot extends AIWithComputationBudget {

    private static final int IDX_WORKER = 0;
    private static final int IDX_LIGHT = 1;
    private static final int IDX_HEAVY = 2;
    private static final int IDX_RANGED = 3;
    private static final int IDX_BASE = 4;
    private static final int IDX_BARRACKS = 5;

    private final UnitTypeTable localUtt;
    private final AStarPathFinding aStar = new AStarPathFinding();

    // [attack/defense][worker/military]
    private final AwareAI[][] strategies;

    private boolean initialized = false;
    private int mapMaxSize;

    private int[] playerUnits = new int[] {0, 0, 0, 0, 0, 0};
    private int[] enemyUnits = new int[] {0, 0, 0, 0, 0, 0};

    private Unit mainBase = null;
    private Unit enemyBase = null;
    private Unit closestEnemy = null;

    private int baseToResources = 9999;
    private int enemyToPlayerBase = 9999;
    private int playerToEnemyBase = 9999;
    private int baseToEnemyBase = 9999;
    private int realBaseToEnemy = -1;

    private int resourceThreshold = 6;
    private int awareness = 4;
    private double strategyPriority = 5.0;

    // smooth decisions to reduce oscillation every frame
    private double attackMomentum = 0.0;
    private double militaryMomentum = 0.0;

    public ChaseBot(AwareAI[][] s, int timeBudget, int iterationsBudget, UnitTypeTable utt) {
        super(timeBudget, iterationsBudget);
        this.localUtt = utt;
        this.strategies = s;
    }

    public ChaseBot(UnitTypeTable utt) {
        this(
                new AwareAI[][] {
                        {new AWorkDefense(utt), new AMilDefense(utt)},
                        {new AWorkRush(utt), new AMilRush(utt)}
                },
                100,
                -1,
                utt
        );
    }

    @Override
    public void reset() {
        initialized = false;
        attackMomentum = 0.0;
        militaryMomentum = 0.0;
        mainBase = null;
        enemyBase = null;
        closestEnemy = null;
    }

    @Override
    public AI clone() {
        return new ChaseBot(TIME_BUDGET, ITERATIONS_BUDGET, localUtt);
    }

    private ChaseBot(int timeBudget, int iterationsBudget, UnitTypeTable utt) {
        this(
                new AwareAI[][] {
                        {new AWorkDefense(utt), new AMilDefense(utt)},
                        {new AWorkRush(utt), new AMilRush(utt)}
                },
                timeBudget,
                iterationsBudget,
                utt
        );
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        parameters.add(new ParameterSpecification("TimeBudget", int.class, 100));
        parameters.add(new ParameterSpecification("IterationsBudget", int.class, -1));
        return parameters;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (!initialized) {
            mapMaxSize = Math.max(gs.getPhysicalGameState().getHeight(), gs.getPhysicalGameState().getWidth());
            strategyPriority = calibrateStrategyPriority(mapMaxSize);
            initialized = true;
        }

        if (!gs.canExecuteAnyAction(player)) {
            return new PlayerAction();
        }

        updateUnitDistribution(player, gs);
        refreshMapDistances(player, gs);

        AwareAI macroStrategy = getMacroStrategy(player, gs);
        calibrateStrategy(macroStrategy, gs, player);
        return macroStrategy.getAction(player, gs);
    }

    private double calibrateStrategyPriority(int mapSize) {
        if (mapSize <= 8) {
            return 10.0;
        }
        if (mapSize <= 16) {
            return 6.0;
        }
        if (mapSize <= 32) {
            return 4.0;
        }
        return 3.0;
    }

    private AwareAI getMacroStrategy(int player, GameState gs) {
        int resources = gs.getPlayer(player).getResources();

        int myMilitary = weightedMilitaryCount(playerUnits);
        int enemyMilitary = weightedMilitaryCount(enemyUnits);
        int militaryLead = myMilitary - enemyMilitary;

        int myEconomy = playerUnits[IDX_WORKER] + (2 * playerUnits[IDX_BASE]) + playerUnits[IDX_BARRACKS];
        int enemyEconomy = enemyUnits[IDX_WORKER] + (2 * enemyUnits[IDX_BASE]) + enemyUnits[IDX_BARRACKS];
        int economyLead = myEconomy - enemyEconomy;

        boolean underThreat = enemyToPlayerBase <= 8;
        boolean enemyCollapsed = enemyUnits[IDX_BASE] == 0 && enemyUnits[IDX_WORKER] <= 2;
        boolean canReachEnemy = realBaseToEnemy >= 0;

        double attackScore =
                (0.9 * militaryLead)
                        + (0.5 * economyLead)
                        + (resources >= 8 ? 1.5 : 0.0)
                        + (canReachEnemy ? 1.0 : -1.0)
                        + (enemyCollapsed ? 3.0 : 0.0)
                        + (underThreat ? -3.0 : 0.0)
                        + (playerUnits[IDX_BASE] == 0 ? -2.0 : 0.0);

        double militaryScore =
                (resources >= resourceThreshold ? 2.0 : -1.0)
                        + (playerUnits[IDX_BARRACKS] > 0 ? 1.5 : -1.5)
                        + (enemyMilitary > 0 ? 1.2 : 0.0)
                        + (mapMaxSize >= 24 ? 0.6 : 0.0)
                        + (playerUnits[IDX_WORKER] <= 1 ? -2.5 : 0.0)
                        + (playerUnits[IDX_BASE] == 0 ? -4.0 : 0.0);

        attackMomentum = 0.70 * attackMomentum + 0.30 * attackScore;
        militaryMomentum = 0.70 * militaryMomentum + 0.30 * militaryScore;

        int attackIdx = attackMomentum >= 0.0 ? 1 : 0;
        int unitModeIdx = militaryMomentum >= 0.0 ? 1 : 0;

        return strategies[attackIdx][unitModeIdx];
    }

    // Tune internal parameters for the selected strategy each frame.
    private void calibrateStrategy(AwareAI strategy, GameState gs, int player) {
        Player p = gs.getPlayer(player);
        int resources = p.getResources();

        int harvestUnits = computeHarvestTarget(gs);
        int totalBarracks = computeBarracksTarget(resources);
        int[] unitProduction = calibrateUnitProduction(gs);

        if (mapMaxSize >= 48 && realBaseToEnemy >= 0) {
            resourceThreshold = 10;
        } else if (playerUnits[IDX_BARRACKS] < totalBarracks) {
            resourceThreshold = mapMaxSize <= 16 ? 5 * (totalBarracks - playerUnits[IDX_BARRACKS]) : 8 * (totalBarracks - playerUnits[IDX_BARRACKS]);
        } else {
            resourceThreshold = 4 + (3 * playerUnits[IDX_BARRACKS]);
        }

        strategy.setnHarvest(harvestUnits);
        strategy.setTotBarracks(totalBarracks);
        strategy.setUnitProduction(unitProduction);
        strategy.setResourceTreshold(resourceThreshold);
        strategy.setPlayerUnits(playerUnits);
        strategy.setEnemyUnits(enemyUnits);
        strategy.setUnitAwareness(awareness);
        strategy.setStrategyPriority(strategyPriority);
    }

    private int computeHarvestTarget(GameState gs) {
        if (mainBase == null) {
            return 0;
        }

        int maxHarvest = Math.max(2, Math.min(6, 1 + (mapMaxSize / 12)));
        int pressure = enemyToPlayerBase <= 10 ? 1 : 0;
        int nearbyResources = countNearbyResources(mainBase, gs.getPhysicalGameState(), 8);

        int target = 2 + (nearbyResources >= 3 ? 1 : 0) + (mapMaxSize >= 32 ? 1 : 0) - pressure;
        target = Math.max(1, Math.min(maxHarvest, target));

        return Math.min(target, playerUnits[IDX_WORKER]);
    }

    private int computeBarracksTarget(int resources) {
        int target = 1;
        if (mapMaxSize >= 24) {
            target++;
        }
        if (resources >= 12 && playerUnits[IDX_WORKER] >= 4 && mapMaxSize >= 32) {
            target++;
        }
        return Math.min(target, 3);
    }

    private int[] calibrateUnitProduction(GameState gs) {
        int[] units = new int[] {1, 1, 1}; // [Light, Heavy, Ranged]

        int myFrontline = playerUnits[IDX_LIGHT] + playerUnits[IDX_HEAVY] + playerUnits[IDX_RANGED];
        int enemyFrontline = enemyUnits[IDX_LIGHT] + enemyUnits[IDX_HEAVY] + enemyUnits[IDX_RANGED];

        if (myFrontline < enemyFrontline) {
            units[IDX_LIGHT - 1] += 1;
            units[IDX_HEAVY - 1] += 1;
        }

        if (enemyUnits[IDX_LIGHT] > enemyUnits[IDX_HEAVY]) {
            units[IDX_HEAVY - 1] += 2; // heavies trade well into light swarms
        }
        if (enemyUnits[IDX_HEAVY] > enemyUnits[IDX_LIGHT]) {
            units[IDX_RANGED - 1] += 2; // ranged punishes slow heavy units
        }
        if (enemyUnits[IDX_RANGED] > 0) {
            units[IDX_LIGHT - 1] += 1; // lights can collapse ranged quickly
        }

        boolean blockedPath = false;
        if (mainBase != null && closestEnemy != null) {
            int d = aStar.findDistToPositionInRange(mainBase, closestEnemy.getPosition(gs.getPhysicalGameState()), 1, gs, gs.getResourceUsage());
            blockedPath = d < 0;
        }

        if (!canReachEnemyBase() || blockedPath || mapMaxSize >= 32) {
            units[IDX_RANGED - 1] += 2;
        }

        if (playerUnits[IDX_BARRACKS] == 0) {
            units[IDX_HEAVY - 1] += 1;
        }

        return units;
    }

    private boolean canReachEnemyBase() {
        return realBaseToEnemy >= 0;
    }

    private int weightedMilitaryCount(int[] units) {
        return units[IDX_WORKER] + (2 * units[IDX_LIGHT]) + (3 * units[IDX_HEAVY]) + (2 * units[IDX_RANGED]);
    }

    private void refreshMapDistances(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        int closestResourceDist = 9999;
        int closestEnemyToBase = 9999;
        int closestAllyToEnemyBase = 9999;
        Unit nearestEnemy = null;

        for (Unit u : pgs.getUnits()) {
            if (mainBase != null && u.getType().isResource) {
                int d = manhattan(mainBase, u);
                if (d < closestResourceDist) {
                    closestResourceDist = d;
                }
            }

            if (mainBase != null && u.getPlayer() >= 0 && u.getPlayer() != player) {
                int d = manhattan(mainBase, u);
                if (d < closestEnemyToBase) {
                    closestEnemyToBase = d;
                    nearestEnemy = u;
                }
            }

            if (enemyBase != null && u.getPlayer() == player) {
                int d = manhattan(enemyBase, u);
                if (d < closestAllyToEnemyBase) {
                    closestAllyToEnemyBase = d;
                }
            }
        }

        closestEnemy = nearestEnemy;
        baseToResources = closestResourceDist;
        enemyToPlayerBase = closestEnemyToBase;
        playerToEnemyBase = closestAllyToEnemyBase;

        if (mainBase != null && enemyBase != null) {
            baseToEnemyBase = manhattan(mainBase, enemyBase);
        } else {
            baseToEnemyBase = 9999;
        }

        if (mainBase != null) {
            realBaseToEnemy = distRealUnitToEnemy(mainBase, gs.getPlayer(player), gs);
        } else {
            realBaseToEnemy = -1;
        }
    }

    private int manhattan(Unit a, Unit b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    private int countNearbyResources(Unit center, PhysicalGameState pgs, int radius) {
        int count = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) {
                int d = Math.abs(center.getX() - u.getX()) + Math.abs(center.getY() - u.getY());
                if (d <= radius) {
                    count++;
                }
            }
        }
        return count;
    }

    private void updateUnitDistribution(int player, GameState gs) {
        playerUnits = new int[] {0, 0, 0, 0, 0, 0};
        enemyUnits = new int[] {0, 0, 0, 0, 0, 0};

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit myBase = null;
        Unit enemyMainBase = null;

        for (Unit u : pgs.getUnits()) {
            int unitIdx = -1;
            String name = u.getType().name;

            if ("Worker".equals(name)) {
                unitIdx = IDX_WORKER;
            } else if ("Light".equals(name)) {
                unitIdx = IDX_LIGHT;
            } else if ("Heavy".equals(name)) {
                unitIdx = IDX_HEAVY;
            } else if ("Ranged".equals(name)) {
                unitIdx = IDX_RANGED;
            } else if ("Base".equals(name)) {
                unitIdx = IDX_BASE;
                if (u.getPlayer() == player && myBase == null) {
                    myBase = u;
                } else if (u.getPlayer() >= 0 && u.getPlayer() != player && enemyMainBase == null) {
                    enemyMainBase = u;
                }
            } else if ("Barracks".equals(name)) {
                unitIdx = IDX_BARRACKS;
            }

            if (unitIdx < 0) {
                continue;
            }

            if (u.getPlayer() == player) {
                playerUnits[unitIdx]++;
            } else if (u.getPlayer() >= 0) {
                enemyUnits[unitIdx]++;
            }
        }

        mainBase = myBase;
        enemyBase = enemyMainBase;
    }

    private int distRealUnitToEnemy(Unit base, Player player, GameState gs) {
        if (base == null) {
            return -1;
        }

        PhysicalGameState pgs = gs.getPhysicalGameState();
        int closestDistance = Integer.MAX_VALUE;

        for (Unit enemy : pgs.getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player.getID()) {
                continue;
            }

            int d = aStar.findDistToPositionInRange(base, enemy.getPosition(pgs), 1, gs, gs.getResourceUsage());
            if (d >= 0 && d < closestDistance) {
                closestDistance = d;
            }
        }

        return closestDistance == Integer.MAX_VALUE ? -1 : closestDistance;
    }
}
