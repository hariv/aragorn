package com.gunrock.aragornbasic;

import java.util.ArrayList;

public class ProfilerResult {
    private ArrayList<String> memoryReadings;
    private ArrayList<String> batteryReadings;
    private String device;
    private boolean useAragorn;

    public ProfilerResult(ArrayList<String> batteryReadings, ArrayList<String> memoryReadings,
                          String device, boolean useAragorn) {
        this.batteryReadings = batteryReadings;
        this.memoryReadings = memoryReadings;
        this.useAragorn = useAragorn;
        this.device = device;
    }
}
