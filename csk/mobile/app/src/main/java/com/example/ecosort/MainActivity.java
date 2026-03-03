package com.example.ecosort;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import java.io.ByteArrayOutputStream;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final int REQUEST_CAMERA = 101;
    private static final int REQUEST_GALLERY = 102;

    private ImageView imageView;
    private TextView resultTextView;
    private Button cameraButton;
    private Button galleryButton;
    private ApiClient apiClient;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        imageView = findViewById(R.id.imageView);
        resultTextView = findViewById(R.id.resultTextView);
        cameraButton = findViewById(R.id.cameraButton);
        galleryButton = findViewById(R.id.galleryButton);

        apiClient = new ApiClient();

        cameraButton.setOnClickListener(v -> checkCameraPermission());
        galleryButton.setOnClickListener(v -> openGallery());
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    REQUEST_CAMERA_PERMISSION
            );
        } else {
            openCamera();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_CAMERA);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_GALLERY);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            Bitmap bitmap = null;

            if (requestCode == REQUEST_CAMERA) {
                bitmap = (Bitmap) data.getExtras().get("data");
            } else if (requestCode == REQUEST_GALLERY) {
                try {
                    Uri uri = data.getData();
                    bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                } catch (Exception e) {
                    e.printStackTrace();
                    Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show();
                    return;
                }
            }

            if (bitmap != null) {
                imageView.setImageBitmap(bitmap);

                classifyImage(bitmap);
            }
        }
    }

    private void classifyImage(Bitmap bitmap) {
        resultTextView.setText("Classifying...");

        Bitmap compressedBitmap = Bitmap.createScaledBitmap(bitmap, 256, 256, true);
        String base64Image = bitmapToBase64(compressedBitmap);

        apiClient.classify(base64Image, new ApiClient.Callback() {
            @Override
            public void onSuccess(ApiClient.ClassifyResponse response) {
                runOnUiThread(() -> {
                    String className = response.className;
                    double confidence = response.confidence * 100;

                    String resultText = String.format(
                            Locale.getDefault(),
                            "Class: %s\nConfidence: %.2f%%",
                            className, confidence
                    );

                    resultTextView.setText(resultText);

                    StringBuilder probs = new StringBuilder("\nProbabilities:\n");
                    for (String key : response.probabilities.keySet()) {
                        double prob = response.probabilities.get(key) * 100;
                        probs.append(String.format("  %s: %.2f%%\n", key, prob));
                    }

                    Toast.makeText(MainActivity.this,
                            className + " (" + String.format("%.1f%%", confidence) + ")",
                            Toast.LENGTH_LONG).show();
                });
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    resultTextView.setText("Error: " + error);
                    Toast.makeText(MainActivity.this, "Classification failed",
                            Toast.LENGTH_SHORT).show();
                });
            }
        });
    }

    private String bitmapToBase64(Bitmap bitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 80, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();
        return android.util.Base64.encodeToString(byteArray, android.util.Base64.NO_WRAP);
    }
}
