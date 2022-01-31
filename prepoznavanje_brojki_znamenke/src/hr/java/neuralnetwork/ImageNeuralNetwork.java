package hr.java.neuralnetwork;

import org.encog.Encog;
import org.encog.EncogError;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.train.strategy.ResetStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.platformspecific.j2se.TrainingDialog;
import org.encog.platformspecific.j2se.data.image.ImageMLData;
import org.encog.platformspecific.j2se.data.image.ImageMLDataSet;
import org.encog.util.downsample.Downsample;
import org.encog.util.downsample.RGBDownsample;
import org.encog.util.downsample.SimpleIntensityDownsample;
import org.encog.util.simple.EncogUtility;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.sql.SQLOutput;
import java.text.DecimalFormat;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.List;

public class ImageNeuralNetwork {
    static class ImagePair{
        private final File file;
        private final int identity;

        public ImagePair(File file, int identity) {
            super();
            this.file = file;
            this.identity = identity;
        }
        public File getFile() {
            return file;
        }
        public int getIdentity() {
            return identity;
        }
    }
    public static void main(String[] args) {
        Instant start = Instant.now();
        if (args.length < 1) {
            System.out.println("Must specify command file. See source for format.");
        } else {
            try {
                final ImageNeuralNetwork program = new ImageNeuralNetwork();
                program.execute(args[0]);
            } catch (final Exception e) {
                e.printStackTrace();
            }
        }
        Encog.getInstance().shutdown();
        DecimalFormat df = new DecimalFormat("0.00");
        System.out.println("Correct trained: " + df.format((trainedScore/6)*100));
        System.out.println("Correct not trained: " + df.format((notTrainedScore/6)*100));
        Instant end = Instant.now();
        System.out.println("Runtime: " + Duration.between(start, end));
    }

    private final List<ImagePair> imageList = new ArrayList<>();
    private final Map<String, String> args = new HashMap<>();
    private final Map<String, Integer> identity2Neuron = new HashMap<>();
    private final Map<Integer, String> neuron2Identity = new HashMap<>();
    private ImageMLDataSet training;
    private String line;
    private int outputCount;
    private int downsampleWidth;
    private int downsampleHeight;
    private BasicNetwork network;
    private Downsample downsample;
    public static double trainedScore = 0, notTrainedScore = 0;

    private int resultCounter = 0;
    public static String[] expectedOutputs = {
        "099/797-370", "099/123-456", "092/925-337", "098/987-654", "098/926-249", "099/584-006",
        "099/012-456", "099/797-370", "099/123-456", "098/987-654", "092/925-337", "098/926-249"
    };

    private int assignIdentity(final String identity) {

        if (this.identity2Neuron.containsKey(identity.toLowerCase())) {
            return this.identity2Neuron.get(identity.toLowerCase());
        }

        final int result = this.outputCount;
        this.identity2Neuron.put(identity.toLowerCase(), result);
        this.neuron2Identity.put(result, identity.toLowerCase());
        this.outputCount++;
        return result;
    }

    public void execute(final String file) throws IOException {
        final FileInputStream fstream = new FileInputStream(file);
        final DataInputStream in = new DataInputStream(fstream);
        final BufferedReader br = new BufferedReader(new InputStreamReader(in));

        while ((this.line = br.readLine()) != null) {
            executeLine();
        }
        in.close();
    }

    private void executeCommand(final String command) throws IOException {
        switch (command) {
            case "input" -> processInput();
            case "createtraining" -> processCreateTraining();
            case "train" -> processTrain();
            case "network" -> processNetwork();
            case "whatis" -> processWhatIs();
        }
    }

    public void executeLine() throws IOException {
        final int index = this.line.indexOf(':');
        if (index == -1) {
            throw new EncogError("Invalid command: " + this.line);
        }

        final String command = this.line.substring(0, index).toLowerCase()
                .trim();
        final String argsStr = this.line.substring(index + 1).trim();
        final StringTokenizer tok = new StringTokenizer(argsStr, ",");
        this.args.clear();
        while (tok.hasMoreTokens()) {
            final String arg = tok.nextToken();
            final int index2 = arg.indexOf(':');
            if (index2 == -1) {
                throw new EncogError("Invalid command: " + this.line);
            }
            final String key = arg.substring(0, index2).toLowerCase().trim();
            final String value = arg.substring(index2 + 1).trim();
            this.args.put(key, value);
        }
        executeCommand(command);
    }

    private String getArg(final String name) {
        final String result = this.args.get(name);
        if (result == null) {
            throw new EncogError("Missing argument " + name + " on line: "
                    + this.line);
        }
        return result;
    }

    private void processCreateTraining() {
        final String strWidth = getArg("width");
        final String strHeight = getArg("height");
        final String strType = getArg("type");

        this.downsampleHeight = Integer.parseInt(strHeight);
        this.downsampleWidth = Integer.parseInt(strWidth);

        if (strType.equals("RGB")) {
            this.downsample = new RGBDownsample();
        } else {
            this.downsample = new SimpleIntensityDownsample();
        }

        this.training = new ImageMLDataSet(this.downsample, true, 1, -1);
        System.out.println("Training set created");
    }

    private void processInput() {
        final String image = getArg("image");
        final String identity = getArg("identity");

        final int idx = assignIdentity(identity);
        final File file = new File(image);

        this.imageList.add(new ImagePair(file, idx));

        System.out.println("Added input image:" + image);
    }

    private void processNetwork() throws IOException {
        System.out.println("Downsampling images...");

        for (final ImagePair pair : this.imageList) {
            final MLData ideal = new BasicMLData(this.outputCount);
            final int idx = pair.getIdentity();
            for (int i = 0; i < this.outputCount; i++) {
                if (i == idx) {
                    ideal.setData(i, 1);
                } else {
                    ideal.setData(i, -1);
                }
            }

            final Image img = ImageIO.read(pair.getFile());
            final ImageMLData data = new ImageMLData(img);
            this.training.add(data, ideal);
        }

        final String strHidden1 = getArg("hidden1");
        final String strHidden2 = getArg("hidden2");

        this.training.downsample(this.downsampleHeight, this.downsampleWidth);

        final int hidden1 = Integer.parseInt(strHidden1);
        final int hidden2 = Integer.parseInt(strHidden2);

        this.network = EncogUtility.simpleFeedForward(this.training
                        .getInputSize(), hidden1, hidden2,
                this.training.getIdealSize(), true);
        System.out.println("Created network: " + this.network.toString());
    }

    private void processTrain() {
        final String strMode = getArg("mode");
        final String strMinutes = getArg("minutes");
        final String strStrategyError = getArg("strategyerror");
        final String strStrategyCycles = getArg("strategycycles");

        System.out.println("Training Beginning... Output patterns="
                + this.outputCount);

        final double strategyError = Double.parseDouble(strStrategyError);
        final int strategyCycles = Integer.parseInt(strStrategyCycles);

        final ResilientPropagation train = new ResilientPropagation(this.network, this.training, 1e-6, 0.9);
        //final QuickPropagation train = new QuickPropagation(this.network, this.training, 2);
        //final ManhattanPropagation train = new ManhattanPropagation(this.network, this.training, 0.01);
        train.addStrategy(new ResetStrategy(strategyError, strategyCycles));

        if (strMode.equalsIgnoreCase("gui")) {
            TrainingDialog.trainDialog(train, this.network, this.training);
        } else {
            final int minutes = Integer.parseInt(strMinutes);
            EncogUtility.trainConsole(train, this.network, this.training,
                    minutes);
        }
        System.out.println("Training Stopped...");
    }

    public void processWhatIs() throws IOException {
        final String filename = getArg("image");
        final File file = new File(filename);
        final BufferedImage img = ImageIO.read(file);
        StringBuilder output = new StringBuilder();
        int currentPixels = 0;
        final int subImageWidth = img.getWidth()/11, subImageHeight = img.getHeight();

        if (img.getWidth() > 190) {
            for (int i = 0; i < 11; i++) {
                final BufferedImage subImg = img.getSubimage(currentPixels, 0, subImageWidth, subImageHeight);
                final ImageMLData input = new ImageMLData(subImg);
                //ImageIO.write(subImg, "jpg", new File("broj"+i+".jpg"));
                currentPixels += subImageWidth;
                input.downsample(this.downsample, true, this.downsampleHeight, this.downsampleWidth, 1, -1);
                final int winner = this.network.winner(input);
                //if(this.neuron2Identity.get(winner).compareTo("/") != 0 && this.neuron2Identity.get(winner).compareTo("-") != 0) {
                    output.append(this.neuron2Identity.get(winner));
                //}
            }
        }
        else{
            final ImageMLData input = new ImageMLData(img);
            input.downsample(this.downsample, true, this.downsampleHeight, this.downsampleWidth, 1, -1);
            final int winner = this.network.winner(input);
            output = new StringBuilder(this.neuron2Identity.get(winner));
        }
        System.out.print("What is: " + filename + ", it seems to be: " + output +
                " Expected: " + expectedOutputs[resultCounter] + " ");
        if(output.compareTo(new StringBuilder(expectedOutputs[resultCounter])) == 0){
            System.out.println(" CORRECT");
            if(resultCounter < 6){
                trainedScore++;
            }
            else{
                notTrainedScore++;
            }
        }
        else{
            System.out.println(" INCORRECT");
        }
        resultCounter++;
    }
}