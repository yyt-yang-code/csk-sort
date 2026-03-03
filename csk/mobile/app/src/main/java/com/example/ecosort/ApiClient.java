package com.example.ecosort;

import okhttp3.*;
import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;


public class ApiClient {
    private static final String BASE_URL = "http://YOUR_SERVER_IP:5000";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    private final OkHttpClient client;
    private final Gson gson;

    public ApiClient() {
        this.client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build();
        this.gson = new Gson();
    }

    /**
     */
    public static class ClassifyRequest {
        @SerializedName("image")
        private String image;

        @SerializedName("format")
        private String format = "base64";

        public ClassifyRequest(String base64Image) {
            this.image = base64Image;
        }
    }

    /**
     */
    public static class ClassifyResponse {
        @SerializedName("class_name")
        public String className;

        @SerializedName("class_id")
        public int classId;

        @SerializedName("confidence")
        public double confidence;

        @SerializedName("probabilities")
        public java.util.Map<String, Double> probabilities;
    }

    /**
     */
    public ClassifyResponse classify(String base64Image, Callback callback) {
        ClassifyRequest request = new ClassifyRequest(base64Image);
        String json = gson.toJson(request);

        RequestBody body = RequestBody.create(json, JSON);
        Request httpRequest = new Request.Builder()
                .url(BASE_URL + "/predict")
                .post(body)
                .build();

        client.newCall(httpRequest).enqueue(new okhttp3.Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                callback.onError(e.getMessage());
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    String responseData = response.body().string();
                    ClassifyResponse classifyResponse = gson.fromJson(
                            responseData,
                            ClassifyResponse.class
                    );
                    callback.onSuccess(classifyResponse);
                } else {
                    callback.onError("Server error: " + response.code());
                }
            }
        });

        return null;
    }

    /**
     */
    public void healthCheck(Callback callback) {
        Request request = new Request.Builder()
                .url(BASE_URL + "/health")
                .get()
                .build();

        client.newCall(request).enqueue(new okhttp3.Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                callback.onError(e.getMessage());
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    callback.onSuccess(null);
                } else {
                    callback.onError("Server error: " + response.code());
                }
            }
        });
    }

    public interface Callback {
        void onSuccess(ClassifyResponse response);
        void onError(String error);
    }
}
