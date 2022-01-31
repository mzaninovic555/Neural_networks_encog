package hr.java.neuralNetwork;

import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import java.text.DecimalFormat;
import java.util.Scanner;

public class Main {
    public static final int BR_INPUT = 18;
    public static final int BR_LANGUAGE = 6;
    public static final int BR_WORDS = 20;
    public static int correctOutputs = 0;

    public static double intToMinMax(double value){
        return (value)/(10);
    }
    public static double[] networkParameters(String word) {
        //length, a, e, i, o, u, r, secondLetter, secondToLastLetter, firstLetter, lastLetter, numberConsanants, mostCommonLetters
        double[] parameters = new double[BR_INPUT];
        parameters[0] = intToMinMax(word.length());
        parameters[1] = intToMinMax(word.chars().filter(ch -> ch == 'a').count());
        parameters[2] = intToMinMax(word.chars().filter(ch -> ch == 'e').count());
        parameters[3] = intToMinMax(word.chars().filter(ch -> ch == 'i').count());
        parameters[4] = intToMinMax(word.chars().filter(ch -> ch == 'o').count());
        parameters[5] = intToMinMax(word.chars().filter(ch -> ch == 'u').count());
        parameters[6] = intToMinMax(word.chars().filter(ch -> ch == 'r').count());
        parameters[7] = (double) (Character.toUpperCase(word.charAt(1)) - 'A' + 1) / 26;
        parameters[8] = (double) (Character.toUpperCase(word.charAt(word.length() - 2)) - 'A' + 1) / 26;
        parameters[9] = (double) (Character.toUpperCase(word.charAt(0)) - 'A' + 1) / 26;
        parameters[10] = (double) (Character.toUpperCase(word.charAt(word.length() - 1)) - 'A' + 1) / 26;
        parameters[11] = intToMinMax(
                word.length() - parameters[1] - parameters[2] - parameters[3] - parameters[4] - parameters[5] - parameters[6]);
        parameters[12] = (double) (Character.toUpperCase(word.charAt(word.length()/2+1)) - 'A' + 1) / 26;
        parameters[13] = intToMinMax(word.chars().filter(ch -> ch == 't').count());
        parameters[14] = intToMinMax(word.chars().filter(ch -> ch == 'n').count());
        parameters[15] = intToMinMax(word.chars().filter(ch -> ch == 's').count());
        parameters[16] = intToMinMax(word.chars().filter(ch -> ch == 'l').count());
        parameters[17] = intToMinMax(word.chars().filter(ch -> ch == 'c').count());


        return parameters;
    }
    public static void formOutput(double[] output){
        String[] dictionaryCroatian = {
                "sir", "caj", "salata", "jogurt", "spageti", "cokolada", "voda",
                "omlet", "vino", "pivo", "jabuka", "grah", "mlijeko", "vanilija",
                "riba", "hamburger", "krema", "puding", "espresso", "pizza"
        };

        for(int i = 0; i < BR_WORDS; i++){
            if(output[i] > 0.49){
                System.out.println(dictionaryCroatian[i]);
                if(IDEAL[i*BR_LANGUAGE][i] == 1){
                    correctOutputs++;
                }
                return;
            }
        }
        System.out.println("KRIVO");
    }
    public static void fillIdeal(){
        int wordCounter = 0;
        for(int i = 0; i < BR_WORDS*BR_LANGUAGE; i++){
            double[] tmpArray = new double[BR_WORDS];
            for(int j = 0; j < BR_WORDS; j++){
                if(wordCounter == j){
                    tmpArray[j] = 1;
                }
                else{
                    tmpArray[j] = 0;
                }
            }
            if((i+1) % BR_LANGUAGE == 0){
                wordCounter++;
            }
            IDEAL[i] = tmpArray;
        }
    }

    static double[][] INPUT = {
            //sir
            networkParameters("syr"),
            networkParameters("syr"),
            networkParameters("ser"),
            networkParameters("syr"),
            networkParameters("syr"),
            networkParameters("sir"),

            //caj
            networkParameters("caj"),
            networkParameters("caj"),
            networkParameters("herbata"),
            networkParameters("chay"),
            networkParameters("chay"),
            networkParameters("caj"),

            //salata
            networkParameters("salat"),
            networkParameters("salat"),
            networkParameters("salatka"),
            networkParameters("salat"),
            networkParameters("salat"),
            networkParameters("solata"),

            //jogurt
            networkParameters("jogurt"),
            networkParameters("jogurt"),
            networkParameters("jogurt"),
            networkParameters("johurt"),
            networkParameters("yogurt"),
            networkParameters("jogurt"),

            //spageti
            networkParameters("spagety"),
            networkParameters("spagety"),
            networkParameters("spaghetti"),
            networkParameters("spahetti"),
            networkParameters("spagetti"),
            networkParameters("spageti"),

            //cokolada
            networkParameters("cokolada"),
            networkParameters("cokolada"),
            networkParameters("cekolada"),
            networkParameters("shokolad"),
            networkParameters("shokolad"),
            networkParameters("cokolada"),

            //voda
            networkParameters("voda"),
            networkParameters("voda"),
            networkParameters("woda"),
            networkParameters("voda"),
            networkParameters("voda"),
            networkParameters("voda"),

            //omlet
            networkParameters("omeleta"),
            networkParameters("omeleta"),
            networkParameters("omlet"),
            networkParameters("omlet"),
            networkParameters("omlet"),
            networkParameters("omleta"),

            //vino
            networkParameters("vino"),
            networkParameters("vino"),
            networkParameters("wino"),
            networkParameters("vyno"),
            networkParameters("vino"),
            networkParameters("vino"),

            //pivo
            networkParameters("pivo"),
            networkParameters("pivo"),
            networkParameters("pivo"),
            networkParameters("piwo"),
            networkParameters("pivo"),
            networkParameters("pivo"),

            //jabuka
            networkParameters("jablko"),
            networkParameters("jablko"),
            networkParameters("jablko"),
            networkParameters("yabluko"),
            networkParameters("yabloko"),
            networkParameters("jabolko"),

            //riza
            networkParameters("fazole"),
            networkParameters("fazula"),
            networkParameters("fasolki"),
            networkParameters("bobi"),
            networkParameters("fasol"),
            networkParameters("fizol"),

            //mlijeko
            networkParameters("mleko"),
            networkParameters("mlieko"),
            networkParameters("mleko"),
            networkParameters("moloko"),
            networkParameters("moloko"),
            networkParameters("mleko"),

            //vanilija
            networkParameters("vanilka"),
            networkParameters("vanilka"),
            networkParameters("wanilia"),
            networkParameters("vanil"),
            networkParameters("vanil"),
            networkParameters("vanilija"),

            //riba
            networkParameters("ryba"),
            networkParameters("ryby"),
            networkParameters("ryba"),
            networkParameters("ryba"),
            networkParameters("ryba"),
            networkParameters("ribe"),

            //hamburger
            networkParameters("hamburger"),
            networkParameters("hamburger"),
            networkParameters("hamburger"),
            networkParameters("hamburher"),
            networkParameters("hamburger"),
            networkParameters("hamburger"),

            //krema
            networkParameters("krem"),
            networkParameters("krem"),
            networkParameters("krem"),
            networkParameters("krem"),
            networkParameters("krem"),
            networkParameters("krema"),

            //puding
            networkParameters("pudink"),
            networkParameters("puding"),
            networkParameters("puding"),
            networkParameters("pudding"),
            networkParameters("pudynh"),
            networkParameters("puding"),

            //espresso
            networkParameters("espresso"),
            networkParameters("espresso"),
            networkParameters("espresso"),
            networkParameters("espresso"),
            networkParameters("espresso"),
            networkParameters("espresso"),

            //pizza
            networkParameters("pizza"),
            networkParameters("pizza"),
            networkParameters("pizza"),
            networkParameters("pitsa"),
            networkParameters("pitstsa"),
            networkParameters("pizza"),
    };
    static double[][] IDEAL = new double[BR_WORDS*BR_LANGUAGE][BR_WORDS];

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        DecimalFormat dfOutput = new DecimalFormat("0.000");
        fillIdeal();

        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, false, BR_INPUT));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, BR_INPUT*3));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, BR_INPUT*3));
        //network.addLayer(new BasicLayer(new ActivationSigmoid(), true, BR_INPUT*2));
        //network.addLayer(new BasicLayer(new ActivationReLU(), true, BR_INPUT));
        network.addLayer(new BasicLayer(new ActivationSoftMax(), false, BR_WORDS));
        network.getStructure().finalizeStructure();
        network.reset();

        MLDataSet trainingSet = new BasicMLDataSet(INPUT, IDEAL);

        final MLTrain train = new Backpropagation(network, trainingSet);
        //final MLTrain train = new QuickPropagation(network, trainingSet, 2);
        //final MLTrain train = new ManhattanPropagation(network, trainingSet, 0.01);
        //final MLTrain train = new ResilientPropagation(network, trainingSet);

        int epochs = 1;
        do{
            train.iteration();
            System.out.println("[INFO] Iterations: " + dfOutput.format(train.getError())
                    + " | Epochs: " + epochs);
            epochs++;
        } while(train.getError() > 0.0005);

        System.out.println("Neural network results: ");

        int i = 0;
        for(MLDataPair pair: trainingSet){
            if(i % BR_LANGUAGE == 0){
                System.out.println();
            }
            final MLData output = network.compute(pair.getInput());
            System.out.print("Actual: ");
            for(int it = 0; it < BR_WORDS; it++){
                System.out.print(dfOutput.format(output.getData(it)) + " ");
            }
            System.out.print(" Ideal: ");
            for(int it = 0; it < BR_WORDS; it++){
                System.out.print(pair.getIdeal().getData(it) + " ");
            }

            formOutput(output.getData());
            i++;
        }
        System.out.println("Broj epoha: " + epochs);
        System.out.println("Tocnost: " + correctOutputs + "/" + BR_LANGUAGE*BR_WORDS);
        System.out.println();
        double[] outputDouble = new double[BR_WORDS];
        String input = "";

        while(input.compareTo("exit") != 0) {
            System.out.print("Enter search phrase: ");
            input = scanner.nextLine();
            network.compute(networkParameters(input), outputDouble);

            System.out.println("Network output: ");
            for(double out : outputDouble){
                System.out.print(dfOutput.format(out) + " ");
            }
            System.out.println();
            formOutput(outputDouble);
        }
    }
}