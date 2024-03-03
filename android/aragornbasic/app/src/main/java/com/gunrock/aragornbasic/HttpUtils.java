package com.gunrock.aragornbasic;

import com.google.gson.Gson;

import java.io.IOException;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class HttpUtils {
    private static String baseUrl = "https://frieza.herokuapp.com/";
    public static final MediaType JSON = MediaType.parse("application/json; charset=utf-8");
    private static OkHttpClient client = new OkHttpClient();
    private static Gson gson = new Gson();

    public static void postAragorn(String endpoint, ProfilerResult profRes) {
        String requestStr = gson.toJson(profRes);
        String url = baseUrl + endpoint;
        RequestBody body = RequestBody.create(requestStr, JSON);
        Request r = new Request.Builder().url(url).post(body).build();

        Response resp = null;

        try {
            resp = client.newCall(r).execute();
        } catch (IOException e) {
            e.printStackTrace();
        }
        finally {
            if (resp != null) {
                resp.close();
            }
        }
    }
}
