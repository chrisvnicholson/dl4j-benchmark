package org.deeplearning4j.CNNMnist;


import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.Models.LeNet;
import org.deeplearning4j.Utils.BenchmarkUtil;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DecimalFormat;

/**
 *
 */
public class Dl4j_LenetMnist {
    private static final Logger log = LoggerFactory.getLogger(Dl4j_LenetMnist.class);

    // values to pass in from command line when compiled, esp running remotely
    @Option(name="--numGPUWorkers",usage="How many workers to use for multiple GPUs.",aliases = "-nGW")
    // Pass in 8 for 4 GPUs
    public int numGPUWorkers = 0;
    @Option(name="--halfPrecision",usage="Apply half precision for GPUs.",aliases = "-ha")
    public boolean half = false;

    protected final int height = 28;
    protected final int width = 28;
    public final int channels = 1;
    public final int numLabels = 10;
    public final int batchSize = 100;
    public final int epochs = 15;
    public final int iterations = 1;
    public final int seed = 42;
    public final int nCores = 32;

    public void run(String[] args) throws Exception {
        long totalTime = System.currentTimeMillis();
        // Parse command line arguments if they exist
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }

        if(numGPUWorkers > 0 && half)
            DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

        long dataLoadTime = System.currentTimeMillis();
        DataSetIterator mnistTrain = new MultipleEpochsIterator(epochs, new MnistDataSetIterator(batchSize,true,12345));
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);
        dataLoadTime = System.currentTimeMillis() - dataLoadTime;

        MultiLayerNetwork network = new LeNet(height, width, channels, numLabels, seed, iterations).init();

        long trainTime = System.currentTimeMillis();
        BenchmarkUtil.train(network, numGPUWorkers, mnistTrain);
        trainTime = System.currentTimeMillis() - trainTime;

        long testTime = System.currentTimeMillis();
        Evaluation eval = network.evaluate(mnistTest);
        log.debug(eval.stats());
        DecimalFormat df = new DecimalFormat("#.####");
        log.info(df.format(eval.accuracy()));
        testTime = System.currentTimeMillis() - testTime;

        totalTime = System.currentTimeMillis() - totalTime ;
        log.info("****************Example finished********************");
        BenchmarkUtil.printTime("Data", dataLoadTime);
        BenchmarkUtil.printTime("Train", trainTime);
        BenchmarkUtil.printTime("Test", testTime);
        BenchmarkUtil.printTime("Total", totalTime);

    }

    public static void main(String[] args) throws Exception {
        new Dl4j_LenetMnist().run(args);
    }

}
