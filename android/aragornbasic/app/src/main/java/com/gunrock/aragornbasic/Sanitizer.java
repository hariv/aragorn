package com.gunrock.aragornbasic;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Sanitizer {
    private static final int numVals = 5;
    private static final int numLabels = 4;
    private static final int trainedImgSize = 416;
    private static final int numAnchors = 3;
    private static final int layerOneSize = 13;
    private static final int layerTwoSize = 26;
    private static final float confidenceThreshold = 0.2f;

    private static final int SANITIZER_INPUT_WIDTH = 416;
    private static final int SANITIZER_INPUT_HEIGHT = 416;

    private static final int SANITIZER_NUM_CHANNELS = 3;
    private static final int SANITIZER_WIDTH_PER_CHANNEL = 4;

    private static final int[] SANITIZER_OUTPUT_LAYER_ONE_DIMS = {1, 13, 13, 27};
    private static final int[] SANITIZER_OUTPUT_LAYER_TWO_DIMS = {1, 26, 26, 27};

    private static final int[][] layerOneAnchors = {{81, 82}, {135, 169}, {344, 319}};
    private static final int[][] layerTwoAnchors = {{10, 14}, {23, 27}, {37, 58}};

    private int numThreads;
    private final Context context;
    private Interpreter interpreter;

    private ByteBuffer imgData;
    private float[][][][] inputImage;

    private int[] intValues;
    private int[] origValues;
    private float[][][][] layerOne;
    private float[][][][] layerTwo;

    public OnDetectionListener listener;

    private Sanitizer(int numThreads, Context context, OnDetectionListener listener) {
        this.numThreads = numThreads;
        this.context = context;
        this.listener = listener;
        setupModel();
    }

    public static Sanitizer create(Context context, OnDetectionListener listener, boolean useAragorn) {
        if (!useAragorn)
            return null;
        return new Sanitizer(4, context, listener);
    }

    private void setupModel() {
        imgData = ByteBuffer.allocateDirect(1 * SANITIZER_INPUT_WIDTH * SANITIZER_INPUT_HEIGHT * SANITIZER_NUM_CHANNELS * SANITIZER_WIDTH_PER_CHANNEL);
        imgData.order(ByteOrder.nativeOrder());

        intValues = new int[SANITIZER_INPUT_WIDTH * SANITIZER_INPUT_HEIGHT];
        origValues = new int[1080 * 1920];
        layerOne = new float[SANITIZER_OUTPUT_LAYER_ONE_DIMS[0]][SANITIZER_OUTPUT_LAYER_ONE_DIMS[1]][SANITIZER_OUTPUT_LAYER_ONE_DIMS[2]][SANITIZER_OUTPUT_LAYER_ONE_DIMS[3]];

        layerTwo = new float[SANITIZER_OUTPUT_LAYER_TWO_DIMS[0]][SANITIZER_OUTPUT_LAYER_TWO_DIMS[1]][SANITIZER_OUTPUT_LAYER_TWO_DIMS[2]][SANITIZER_OUTPUT_LAYER_TWO_DIMS[3]];

        /*Log.d("setup0", String.valueOf(layerTwo[0][1][2][0]));
        Log.d("setup1", String.valueOf(layerTwo[0][3][4][1]));
        Log.d("setup2", String.valueOf(layerTwo[0][5][6][2]));
        Log.d("setup3", String.valueOf(layerTwo[0][7][8][3]));
        Log.d("setup4", String.valueOf(layerTwo[0][9][10][4]));
        Log.d("setup5", String.valueOf(layerTwo[0][11][12][5]));*/

        File f = new File(context.getCacheDir() + "/sanitizer.tflite");
        if (!f.exists()) {
            try {
                InputStream is = context.getAssets().open("sanitizer.tflite");
                OutputStream fos = new FileOutputStream(f);
                byte[] buffer = new byte[1024];
                int length = 0;

                while ((length=is.read(buffer)) > 0) {
                    fos.write(buffer, 0, length);
                }
                fos.close();
                is.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(numThreads);
        interpreter = new Interpreter(f, options);
    }

    private float sigmoid(float x) {
        return (float) (1.0f / (1.0f + Math.exp(-x)));
    }

    private float[] softmax(float[] input, int size) {
        int i;
        float m, sum, constant;
        float[] res = new float[input.length];

        m = -Float.MAX_VALUE;
        for (i = 0; i < size; i++) {
            if (m < input[i]) {
                m = input[i];
            }
        }

        sum = 0.0f;
        for (i = 0; i < size; ++i) {
            sum += Math.exp(input[i] - m);
        }

        constant = (float) (m + Math.log(sum));
        for (i = 0; i < size; i++) {
            res[i] = (float) Math.exp(input[i] - constant);
        }
        return res;
    }

    private int getMaxElementIndex(float[] confidenceClasses) {
        int maxElementIndex = 0;
        for (int i = 0; i < confidenceClasses.length; i++) {
            maxElementIndex = (confidenceClasses[i] > confidenceClasses[maxElementIndex]) ? i : maxElementIndex;
        }
        return maxElementIndex;
    }

    private float[][] processYoloLayer(float[][][][] layer, int layerSize, int[][] anchors, float[][] res) {
        float[][] results = res;

        for (int p = 0; p < layer.length; p++) {
            for (int q = 0; q < layer[p].length; q++) {
                for (int r = 0; r < layer[p][q].length; r++) {
                    for (int s = 0; s < numAnchors; s++) {
                        int offset = (numLabels + 5) * s;
                        float confidence = sigmoid(layer[p][q][r][offset + 4]);
                        float[] confidenceClasses = new float[numLabels];
                        for (int c = 0; c < numLabels; c++) {
                            confidenceClasses[c] = layer[p][q][r][offset + 5 + c];
                        }
                        confidenceClasses = softmax(confidenceClasses, numLabels);
                        int objectId = getMaxElementIndex(confidenceClasses);
                        float maxConf = confidenceClasses[objectId];
                        confidence = confidence * maxConf;

                        if (confidence > confidenceThreshold) {
                            float x = (r + sigmoid(layer[p][q][r][offset])) / layerSize;
                            float y = (q + sigmoid(layer[p][q][r][offset + 1])) / layerSize;
                            float w = (float) (Math.exp(layer[p][q][r][offset + 2]) * anchors[s][0] / trainedImgSize);
                            float h = (float) (Math.exp(layer[p][q][r][offset + 3]) * anchors[2][1] / trainedImgSize);

                            if (results[objectId][4] < confidence) {
                                results[objectId][0] = x - w / 2;
                                results[objectId][1] = y - h / 2;
                                results[objectId][2] = x + w / 2;
                                results[objectId][3] = y + h / 2;
                                results[objectId][4] = confidence;
                            }
                        }
                        confidenceClasses = null;
                    }
                }
            }
        }

        return results;
    }

    public void sanitize(Bitmap image) {

        if (interpreter == null) {
            setupModel();
        }

        int imageWidth = image.getWidth(), imageHeight = image.getHeight();

        //image.getPixels(origValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
        //image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
        Bitmap bitmap = Bitmap.createScaledBitmap(image, SANITIZER_INPUT_WIDTH, SANITIZER_INPUT_HEIGHT, false);

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        inputImage = new float[1][SANITIZER_INPUT_HEIGHT][SANITIZER_INPUT_WIDTH][SANITIZER_NUM_CHANNELS];
        ArrayList<Integer> redPixelList = new ArrayList<>();

        /*for (int i = 0; i < imageHeight; ++i) {
            for (int j = 0; j < imageWidth; ++j) {
                int pixelValue = origValues[i * imageWidth + j];
                redPixelList.add((pixelValue >> 16) & 0xFF);
            }
        }*/

        System.out.println("testhari " + (byte)(80.0/255.0));
        for (int i = 0; i < SANITIZER_INPUT_HEIGHT; ++i) {
            for (int j = 0; j < SANITIZER_INPUT_WIDTH; ++j) {
                int pixelValue = intValues[i * SANITIZER_INPUT_HEIGHT + j];
                //int pixelValue = 1;
                float redPixel = (float) ((pixelValue >> 16) & 0xFF);
                float greenPixel = (float) ((pixelValue >> 8) & 0xFF);
                float bluePixel = (float) (pixelValue & 0xFF);
                redPixelList.add((pixelValue >> 16) & 0xFF);
                //imgData.put((byte) (redPixel));
                //imgData.put((byte) (greenPixel));
                //imgData.put((byte) (bluePixel));
                imgData.put((byte) (redPixel / 255.0));
                imgData.put((byte) (greenPixel / 255.0));
                imgData.put((byte) (bluePixel / 255.0));
                inputImage[0][i][j][0] = (float) (redPixel / 255.0);
                inputImage[0][i][j][1] = (float) (greenPixel / 255.0);
                inputImage[0][i][j][2] = (float) (bluePixel / 255.0);
            }
        }


        //Log.d("redPixesize", String.valueOf(redPixelList.size()));
        //Log.d("redpixels", redPixelList.subList(86528, 86944).toString());
        //Object[] inputArray = {imgData};
        Object[] inputArray = {inputImage};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, layerOne);
        outputMap.put(1, layerTwo);

        /*Log.d("pre0", String.valueOf(layerTwo[0][1][2][0]));
        Log.d("pre1", String.valueOf(layerTwo[0][3][4][1]));
        Log.d("pre2", String.valueOf(layerTwo[0][5][6][2]));
        Log.d("pre3", String.valueOf(layerTwo[0][7][8][3]));
        Log.d("pre4", String.valueOf(layerTwo[0][9][10][4]));
        Log.d("pre5", String.valueOf(layerTwo[0][11][12][5]));*/
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

        float[][] res = new float[numLabels][numVals];

        for (float[] row : res) {
            Arrays.fill(row, -Float.MAX_VALUE);
        }
        //Log.d("debugres", String.valueOf(res.length));
        //Log.d("debugres", String.valueOf(res[0].length));

        Log.d("test0", String.valueOf(layerTwo[0][1][2][0]));
        Log.d("test1", String.valueOf(layerTwo[0][3][4][1]));
        Log.d("test2", String.valueOf(layerTwo[0][5][6][2]));
        Log.d("test3", String.valueOf(layerTwo[0][7][8][3]));
        Log.d("test4", String.valueOf(layerTwo[0][9][10][4]));
        Log.d("test5", String.valueOf(layerTwo[0][11][12][5]));
        //Log.d("test6", String.valueOf(layerOne[0][0][0][6]));

        res = processYoloLayer(layerOne, layerOneSize, layerOneAnchors, res);
        res = processYoloLayer(layerTwo, layerTwoSize, layerTwoAnchors, res);

        Detection d = new Detection(res[1][0] * imageWidth, res[1][1] * imageHeight,
                res[1][2] * imageWidth, res[1][3] * imageHeight);

        Log.d("left", String.valueOf(res[1][0] * imageWidth));
        Log.d("top", String.valueOf(res[1][1] * imageHeight));
        Log.d("right", String.valueOf(res[1][2] * imageWidth));
        Log.d("bottom", String.valueOf(res[1][3] * imageHeight));

        //Detection d = new Detection(56.49f, 614.096f, 1024.93f, 1204.97f);
        listener.onResults(d);
    }
}
