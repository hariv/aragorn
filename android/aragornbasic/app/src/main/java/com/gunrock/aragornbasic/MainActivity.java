package com.gunrock.aragornbasic;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;

import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Size;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity implements OnDetectionListener{
    private int REQUEST_CODE_PERMISSIONS = 1001;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.INTERNET"};
    private Bitmap bitmapBuffer;
    private ExecutorService cameraExecutor;
    private Sanitizer sanitizer;
    private final Object task = new Object();

    private Boolean useAragorn = true;
    private OverlayView overlayView;
    PreviewView mPreviewView;

    public static Bitmap removeAlpha(Bitmap sourceBitmap) {
        // Create a new Bitmap with RGB_565 configuration
        Bitmap resultBitmap = Bitmap.createBitmap(sourceBitmap.getWidth(), sourceBitmap.getHeight(), Bitmap.Config.RGB_565);

        // Create a Canvas to draw the sourceBitmap onto the resultBitmap
        Canvas canvas = new Canvas(resultBitmap);

        // Set background color to white (or any color you prefer)
        canvas.drawColor(Color.WHITE);

        // Draw the sourceBitmap onto the resultBitmap using a Paint with DITHER_FLAG
        Paint paint = new Paint();
        paint.setDither(true);
        canvas.drawBitmap(sourceBitmap, 0, 0, paint);

        return resultBitmap;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mPreviewView = findViewById(R.id.previewView);
        overlayView = (OverlayView) findViewById(R.id.overlay);

        cameraExecutor = Executors.newSingleThreadExecutor();

        sanitizer = Sanitizer.create(getBaseContext(), this, useAragorn);

        /*AssetManager assetManager = getApplicationContext().getAssets();
        try {
            InputStream inputStream = assetManager.open("frame.jpg");
            bitmapBuffer = BitmapFactory.decodeStream(inputStream);
            inputStream.close();


            //bitmapBuffer = removeAlpha(bitmapBuffer);
            if (bitmapBuffer.getConfig() == Bitmap.Config.ARGB_8888 || bitmapBuffer.hasAlpha()) {
                Log.d("alpha", "bitmap has alpha");
            }
            else {
                Log.d("alpha", "bitmap has no alpha");
            }

            /*Bitmap bitmap = Bitmap.createScaledBitmap(bitmapBuffer, 416, 416, false);
            int[] redChannel = extractRedChannel(bitmap);

            for (int i = 0 ; i < 416; i++) {
                int[] row = new int[416];
                for (int j = 0; j < 416; j++) {
                    int pixelValue = redChannel[i * 416 + j];
                    row[j] = pixelValue;
                }
                Log.d("row_" + String.valueOf(i), Arrays.toString(row));
            }*//*

        } catch (IOException e) {
            e.printStackTrace();
        }*/

        if(allPermissionsGranted()){
            startCamera();
        } else{
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera() {

        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {

                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    bindPreview(cameraProvider);

                } catch (ExecutionException | InterruptedException e) {
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {

        Preview preview = new Preview.Builder()
                .setTargetResolution(new Size(1080, 1920))
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                .setTargetResolution(new Size(1080, 1920))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();

        imageAnalysis.setAnalyzer(cameraExecutor, image -> {

            if (bitmapBuffer == null) {
                bitmapBuffer = Bitmap.createBitmap(
                        image.getWidth(),
                        image.getHeight(),
                        Bitmap.Config.ARGB_8888);
            }
            processFrame(image);
        });

        preview.setSurfaceProvider(mPreviewView.getSurfaceProvider());

        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis);
    }

    private void processFrame(@NonNull ImageProxy image) {
        bitmapBuffer.copyPixelsFromBuffer(image.getPlanes()[0].getBuffer());
        image.close();

        synchronized (task) {
            if (sanitizer != null) {
                sanitizer.sanitize(bitmapBuffer);
            }
        }
    }
    private boolean allPermissionsGranted(){
        for(String permission : REQUIRED_PERMISSIONS){
            if(ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED){
                return false;
            }
        }
        return true;
    }

    public static int[] extractRedChannel(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        int[] redChannel = new int[width * height];
        //int[] redChannel = new int[height * width];

        /*for (int y = 0; y < width; y++) {
            for (int x = 0; x < height; x++) {
                int pixel = bitmap.getPixel(y, x);

                int red = Color.red(pixel);

                redChannel[y * height + x] = red;
            }
        }*/
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = bitmap.getPixel(x, y);

                // Extract red component using Color class
                int red = Color.red(pixel);

                // Store the red value in the red channel array
                redChannel[y * width + x] = red;
            }
        }

        return redChannel;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if(requestCode == REQUEST_CODE_PERMISSIONS){
            if(allPermissionsGranted()){
                startCamera();
            } else{
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }

    @Override
    public void onResults(Detection detection) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                overlayView.setBB(detection.left, detection.top, detection.right, detection.bottom);
                overlayView.invalidate();
            }
        });
    }
}