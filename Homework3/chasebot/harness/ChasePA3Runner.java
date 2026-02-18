package tests;

import ai.core.AI;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.UnitAction;
import rts.units.Unit;
import rts.units.UnitTypeTable;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.lang.reflect.Constructor;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

public class ChasePA3Runner {

    static class OpponentResult {
        String name;
        String className;
        boolean available;
        String reason;
        int wins;
        int losses;
        int draws;
        int gamesPlayed;

        OpponentResult(String name) {
            this.name = name;
            this.className = "";
            this.available = false;
            this.reason = "uninitialized";
        }
    }

    static class MatchOutcome {
        int wins;
        int losses;
        int draws;
        int gamesPlayed;
    }

    static class AvailableOpponent {
        String name;
        String className;

        AvailableOpponent(String name, String className) {
            this.name = name;
            this.className = className;
        }
    }

    static class GameResult {
        String opponent;
        String opponentClass;
        int gameIndex;
        boolean chaseAsP0;
        String winnerLabel;
        int perspectiveWinner;
        int cycles;
        int frames;
        String framePrefix;
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> a = parseArgs(args);

        String chaseClass = a.getOrDefault("chase-class", "ai.chasebot.ChaseBot");
        String mapPathArg = a.getOrDefault("map", "");
        int games = Integer.parseInt(a.getOrDefault("games", "6"));
        int maxTotalGames = Integer.parseInt(a.getOrDefault("max-total-games", "-1"));
        int maxCycles = Integer.parseInt(a.getOrDefault("max-cycles", "5000"));
        int frameEvery = Integer.parseInt(a.getOrDefault("frame-every", "110"));
        String framesDir = a.getOrDefault("frames-dir", "../../Homework3/chasebot/report/images/match1");
        String resultsPath = a.getOrDefault("results", "../../Homework3/chasebot/report/benchmark-results.tsv");
        boolean captureAll = Boolean.parseBoolean(a.getOrDefault("capture-all", "true"));

        UnitTypeTable utt = new UnitTypeTable();
        String mapPath = resolveMapPath(mapPathArg);

        Path frameRoot = Paths.get(framesDir);
        Files.createDirectories(frameRoot);

        Map<String, String[]> candidates = new LinkedHashMap<>();
        candidates.put("Random", new String[] {"ai.RandomBiasedAI", "ai.RandomAI", "ai.abstraction.RandomBiasedAI"});
        candidates.put("WorkerRush", new String[] {"ai.abstraction.WorkerRush"});
        candidates.put("LightRush", new String[] {"ai.abstraction.LightRush"});
        candidates.put("NaiveMCTS", new String[] {"ai.mcts.naivemcts.NaiveMCTS"});
        candidates.put("Mayari", new String[] {"ai.mayari.Mayari", "ai.mayari.MayariBot", "mayari.MayariBot", "ai.mayaribot.MayariBot"});
        candidates.put("Coac", new String[] {"ai.coac.CoacAI", "ai.coac.Coac", "coac.CoacAI", "ai.competition.coac.CoacAI"});

        List<OpponentResult> allResults = new ArrayList<>();
        Map<String, OpponentResult> resultByName = new LinkedHashMap<>();
        List<AvailableOpponent> availableOpps = new ArrayList<>();
        List<GameResult> gameResults = new ArrayList<>();

        for (Map.Entry<String, String[]> entry : candidates.entrySet()) {
            String oppName = entry.getKey();
            OpponentResult result = new OpponentResult(oppName);

            String resolved = resolveClass(entry.getValue(), utt);
            if (resolved == null) {
                result.available = false;
                result.reason = "class not found or constructor unsupported";
                allResults.add(result);
                resultByName.put(oppName, result);
                continue;
            }

            result.className = resolved;
            result.available = true;
            result.reason = "";
            allResults.add(result);
            resultByName.put(oppName, result);
            availableOpps.add(new AvailableOpponent(oppName, resolved));
        }

        int totalBudget = maxTotalGames > 0 ? maxTotalGames : games * availableOpps.size();
        if (totalBudget < 0) totalBudget = 0;

        int availableCount = availableOpps.size();
        int basePerOpp = (availableCount > 0) ? Math.min(games, totalBudget / availableCount) : 0;
        int rem = (availableCount > 0) ? Math.min(availableCount * games, totalBudget) - (basePerOpp * availableCount) : 0;
        int[] planned = new int[availableCount];

        for (int i = 0; i < availableCount; i++) {
            int quota = basePerOpp + (i < rem ? 1 : 0);
            planned[i] = Math.min(games, Math.max(0, quota));
        }

        int plannedTotal = 0;
        for (int p : planned) plannedTotal += p;

        if (plannedTotal < totalBudget && availableCount > 0) {
            int need = Math.min(totalBudget, availableCount * games) - plannedTotal;
            int idx = 0;
            while (need > 0) {
                if (planned[idx] < games) {
                    planned[idx]++;
                    need--;
                }
                idx = (idx + 1) % availableCount;
            }
        }

        for (int i = 0; i < availableCount; i++) {
            AvailableOpponent ao = availableOpps.get(i);
            MatchOutcome out = runOpponent(
                    chaseClass,
                    ao.className,
                    mapPath,
                    planned[i],
                    maxCycles,
                    frameEvery,
                    frameRoot,
                    captureAll,
                    ao.name,
                    utt,
                    gameResults
            );

            OpponentResult result = resultByName.get(ao.name);
            if (result != null) {
                result.wins = out.wins;
                result.losses = out.losses;
                result.draws = out.draws;
                result.gamesPlayed = out.gamesPlayed;
            }
        }

        writeResults(resultsPath, mapPath, games, maxCycles, chaseClass, allResults, gameResults);
    }

    private static MatchOutcome runOpponent(
            String chaseClass,
            String opponentClass,
            String mapPath,
            int games,
            int maxCycles,
            int frameEvery,
            Path frameRoot,
            boolean captureFrames,
            String opponentName,
            UnitTypeTable utt,
            List<GameResult> gameResults
    ) throws Exception {
        MatchOutcome out = new MatchOutcome();

        for (int g = 0; g < games; g++) {
            boolean chaseAsP0 = (g % 2 == 0);
            AI chase = createAI(chaseClass, utt);
            AI opp = createAI(opponentClass, utt);

            GameResult game = runSingleGame(
                    chase,
                    opp,
                    chaseAsP0,
                    mapPath,
                    maxCycles,
                    frameEvery,
                    frameRoot,
                    captureFrames,
                    opponentName,
                    opponentClass,
                    g
            );

            gameResults.add(game);

            if (game.perspectiveWinner > 0) {
                out.wins++;
            } else if (game.perspectiveWinner < 0) {
                out.losses++;
            } else {
                out.draws++;
            }
            out.gamesPlayed++;
        }

        return out;
    }

    private static GameResult runSingleGame(
            AI chase,
            AI opp,
            boolean chaseAsP0,
            String mapPath,
            int maxCycles,
            int frameEvery,
            Path frameRoot,
            boolean captureFrames,
            String opponentName,
            String opponentClass,
            int gameIdx
    ) throws Exception {
        UnitTypeTable utt = new UnitTypeTable();
        PhysicalGameState pgs = PhysicalGameState.load(mapPath, utt);
        GameState gs = new GameState(pgs, utt);

        AI p0 = chaseAsP0 ? chase : opp;
        AI p1 = chaseAsP0 ? opp : chase;

        boolean gameover = false;
        int nextFrame = 0;
        int frameCounter = 1;

        String framePrefix = sanitize(opponentName) + "_g" + String.format(Locale.US, "%02d", gameIdx + 1) + "_";

        while (!gameover && gs.getTime() < maxCycles) {
            if (captureFrames && frameEvery > 0 && gs.getTime() >= nextFrame) {
                String f = framePrefix + "t" + String.format(Locale.US, "%05d", gs.getTime())
                        + "_f" + String.format(Locale.US, "%05d", frameCounter)
                        + ".png";
                saveFrame(gs, frameRoot.resolve(f));
                nextFrame += frameEvery;
                frameCounter++;
            }

            if (gs.isComplete()) {
                gameover = gs.cycle();
                continue;
            }

            PlayerAction pa0 = p0.getAction(0, gs);
            PlayerAction pa1 = p1.getAction(1, gs);

            if (pa0 == null) pa0 = new PlayerAction();
            if (pa1 == null) pa1 = new PlayerAction();

            gs.issueSafe(pa0);
            gs.issueSafe(pa1);

            gameover = gs.cycle();
        }

        int winner = gs.winner();

        try {
            p0.gameOver(winner);
            p1.gameOver(winner);
        } catch (Exception ignored) {
        }

        GameResult result = new GameResult();
        result.opponent = opponentName;
        result.opponentClass = opponentClass;
        result.gameIndex = gameIdx + 1;
        result.chaseAsP0 = chaseAsP0;
        result.cycles = gs.getTime();
        result.frames = Math.max(0, frameCounter - 1);
        result.framePrefix = framePrefix;

        if (winner == -1) {
            result.winnerLabel = "draw";
            result.perspectiveWinner = 0;
            return result;
        }

        boolean chaseWon = (winner == 0 && chaseAsP0) || (winner == 1 && !chaseAsP0);
        result.perspectiveWinner = chaseWon ? 1 : -1;
        result.winnerLabel = chaseWon ? "chase" : "opponent";
        return result;
    }

    private static void saveFrame(GameState gs, Path outPath) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int cell = 24;
        int header = 28;
        int widthPx = Math.max(1, pgs.getWidth() * cell);
        int heightPx = Math.max(1, header + pgs.getHeight() * cell);

        BufferedImage image = new BufferedImage(widthPx, heightPx, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g.setColor(Color.WHITE);
        g.fillRect(0, 0, widthPx, heightPx);

        g.setColor(new Color(242, 242, 242));
        g.fillRect(0, header, widthPx, heightPx - header);

        g.setColor(Color.BLACK);
        g.setFont(new Font("SansSerif", Font.BOLD, 12));
        g.drawString("t=" + gs.getTime(), 8, 18);

        for (int y = 0; y < pgs.getHeight(); y++) {
            for (int x = 0; x < pgs.getWidth(); x++) {
                int px = x * cell;
                int py = header + y * cell;

                if (pgs.getTerrain(x, y) == PhysicalGameState.TERRAIN_WALL) {
                    g.setColor(new Color(60, 60, 60));
                    g.fillRect(px, py, cell, cell);
                }

                g.setColor(new Color(205, 205, 205));
                g.drawRect(px, py, cell, cell);
            }
        }

        for (Unit u : pgs.getUnits()) {
            int px = u.getX() * cell;
            int py = header + u.getY() * cell;

            if (u.getType().isResource) {
                g.setColor(new Color(40, 150, 40));
                g.fillOval(px + 4, py + 4, cell - 8, cell - 8);
                continue;
            }

            Color c;
            if (u.getPlayer() == 0) c = new Color(45, 110, 220);
            else if (u.getPlayer() == 1) c = new Color(220, 60, 60);
            else c = new Color(90, 90, 90);

            if (u.getType().isStockpile || "Base".equals(u.getType().name)) {
                g.setColor(c);
                g.fillRect(px + 2, py + 2, cell - 4, cell - 4);
            } else if ("Barracks".equals(u.getType().name)) {
                g.setColor(c.darker());
                g.fillRect(px + 3, py + 3, cell - 6, cell - 6);
            } else {
                g.setColor(c);
                g.fillOval(px + 3, py + 3, cell - 6, cell - 6);
            }

            g.setColor(Color.WHITE);
            g.setFont(new Font("SansSerif", Font.BOLD, 10));
            String letter = u.getType().name.substring(0, 1).toUpperCase(Locale.US);
            g.drawString(letter, px + 8, py + 15);
        }

        g.dispose();
        Files.createDirectories(outPath.getParent());
        ImageIO.write(image, "png", outPath.toFile());
    }

    private static AI createAI(String className, UnitTypeTable utt) throws Exception {
        Class<?> cls = Class.forName(className);
        if (!AI.class.isAssignableFrom(cls)) {
            throw new IllegalArgumentException(className + " does not implement ai.core.AI");
        }

        Constructor<?>[] ctors = cls.getConstructors();

        Constructor<?> directUtt = null;
        Constructor<?> noArg = null;
        List<Constructor<?>> generic = new ArrayList<>();

        for (Constructor<?> c : ctors) {
            Class<?>[] p = c.getParameterTypes();
            if (p.length == 1 && p[0].equals(UnitTypeTable.class)) {
                directUtt = c;
            } else if (p.length == 0) {
                noArg = c;
            } else {
                generic.add(c);
            }
        }

        if (directUtt != null) {
            return (AI) directUtt.newInstance(utt);
        }

        if (noArg != null) {
            return (AI) noArg.newInstance();
        }

        generic.sort((a, b) -> Integer.compare(a.getParameterCount(), b.getParameterCount()));

        for (Constructor<?> c : generic) {
            Object[] args = buildArgs(c.getParameterTypes(), utt);
            if (args != null) {
                try {
                    return (AI) c.newInstance(args);
                } catch (Exception ignored) {
                }
            }
        }

        throw new IllegalArgumentException("No supported constructor for " + className);
    }

    private static Object[] buildArgs(Class<?>[] types, UnitTypeTable utt) {
        Object[] args = new Object[types.length];
        for (int i = 0; i < types.length; i++) {
            Class<?> t = types[i];
            if (t.equals(UnitTypeTable.class)) args[i] = utt;
            else if (t.equals(int.class) || t.equals(Integer.class)) args[i] = 100;
            else if (t.equals(long.class) || t.equals(Long.class)) args[i] = 100L;
            else if (t.equals(float.class) || t.equals(Float.class)) args[i] = 0.25f;
            else if (t.equals(double.class) || t.equals(Double.class)) args[i] = 0.25d;
            else if (t.equals(boolean.class) || t.equals(Boolean.class)) args[i] = false;
            else if (t.equals(String.class)) args[i] = "";
            else return null;
        }
        return args;
    }

    private static String resolveClass(String[] classCandidates, UnitTypeTable utt) {
        for (String c : classCandidates) {
            try {
                AI ai = createAI(c, utt);
                if (ai != null) {
                    return c;
                }
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    private static String resolveMapPath(String requested) throws Exception {
        if (requested != null && !requested.isEmpty() && Files.exists(Paths.get(requested))) {
            return requested;
        }

        String[] candidates = new String[] {
                "maps/8x8/basesWorkers8x8.xml",
                "maps/16x16/basesWorkers16x16.xml",
                "maps/24x24/basesWorkers24x24.xml",
                "maps/32x32/basesWorkers32x32.xml"
        };

        for (String c : candidates) {
            if (Files.exists(Paths.get(c))) {
                return c;
            }
        }

        try (Stream<Path> stream = Files.walk(Paths.get("maps"))) {
            Path found = stream
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase(Locale.US).endsWith(".xml"))
                    .filter(p -> p.getFileName().toString().toLowerCase(Locale.US).contains("basesworkers"))
                    .findFirst()
                    .orElse(null);
            if (found != null) {
                return found.toString();
            }
        }

        try (Stream<Path> stream = Files.walk(Paths.get("maps"))) {
            Path found = stream
                    .filter(Files::isRegularFile)
                    .filter(p -> p.toString().toLowerCase(Locale.US).endsWith(".xml"))
                    .findFirst()
                    .orElse(null);
            if (found != null) {
                return found.toString();
            }
        }

        throw new IllegalStateException("Could not find any map XML in maps/");
    }

    private static void writeResults(
            String path,
            String mapPath,
            int games,
            int maxCycles,
            String chaseClass,
            List<OpponentResult> rows,
            List<GameResult> gameRows
    ) throws Exception {
        Path out = Paths.get(path);
        Files.createDirectories(out.getParent());

        try (BufferedWriter w = new BufferedWriter(new FileWriter(out.toFile()))) {
            w.write("META\tmap\t" + mapPath + "\tgames\t" + games + "\tmax_cycles\t" + maxCycles + "\tchase_class\t" + chaseClass);
            w.newLine();

            for (OpponentResult r : rows) {
                w.write("OPP\t" + r.name
                        + "\tavailable\t" + r.available
                        + "\tclass\t" + nullToEmpty(r.className)
                        + "\tgames_played\t" + r.gamesPlayed
                        + "\twins\t" + r.wins
                        + "\tlosses\t" + r.losses
                        + "\tdraws\t" + r.draws
                        + "\treason\t" + nullToEmpty(r.reason)
                );
                w.newLine();
            }

            for (GameResult g : gameRows) {
                w.write("GAME\t" + g.opponent
                        + "\tclass\t" + nullToEmpty(g.opponentClass)
                        + "\tgame\t" + g.gameIndex
                        + "\tchase_as_p0\t" + g.chaseAsP0
                        + "\twinner\t" + g.winnerLabel
                        + "\tperspective\t" + g.perspectiveWinner
                        + "\tcycles\t" + g.cycles
                        + "\tframes\t" + g.frames
                        + "\tframe_prefix\t" + nullToEmpty(g.framePrefix)
                );
                w.newLine();
            }
        }
    }

    private static String nullToEmpty(String s) {
        return s == null ? "" : s;
    }

    private static String sanitize(String s) {
        return s.replaceAll("[^A-Za-z0-9._-]", "_");
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> out = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String k = args[i];
            if (!k.startsWith("--")) continue;
            if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                out.put(k.substring(2), args[i + 1]);
                i++;
            } else {
                out.put(k.substring(2), "true");
            }
        }
        return out;
    }
}
