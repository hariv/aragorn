package com.gunrock.aragornbasic;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class Profiler {
    private final long duration;
    private final long sleepDuration;
    private final Boolean useAragorn;
    private final String device;
    private final String memoryUsageFilename = "/proc/meminfo";
    private final File memoryUsageFile = new File(memoryUsageFilename);
    public Context mContext;

    public Profiler(long duration, long sleepDuration, Boolean useAragorn, String device, Context mContext) {
        this.duration = duration;
        this.sleepDuration = sleepDuration;
        this.mContext = mContext;
        this.useAragorn = useAragorn;
        this.device = device;
    }

    private String readMemoryUsage(File memoryUsageFile) {
        String memTotalString = "";
        String memFreeString = "";
        String memUsageString = "";
        int memTotalTargetLine = 0;
        int memFreeTargetLine = 1;
        Scanner scanner = null;

        try {
            scanner = new Scanner(new FileInputStream(memoryUsageFile));
            int currentLine = 0;
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (currentLine == memTotalTargetLine)
                    memTotalString = line;
                else if (currentLine == memFreeTargetLine)
                    memFreeString = line;
                currentLine++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        if (memTotalString != "" && memFreeString != "") {
            long memTotal = Long.parseLong(memTotalString.split("\\s+")[1]);
            long memFree = Long.parseLong(memFreeString.split("\\s+")[1]);
            memUsageString = String.valueOf(memTotal - memFree);
        }
        return memUsageString;
    }

    public String readBatteryUsage() {
        IntentFilter ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        Intent batteryStatus = mContext.registerReceiver(null, ifilter);

        int level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
        int scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
        float batteryPct = level / (float) scale;

        return String.valueOf(batteryPct);
    }

    public ProfilerResult profile() {
        ProfilerResult result = null;
        long startTime = System.currentTimeMillis();
        ArrayList<String> memoryReadings = new ArrayList<String>();
        ArrayList<String> batteryReadings = new ArrayList<String>();

        while (true) {
            String memUsage = readMemoryUsage(memoryUsageFile);
            if (memUsage != "") {
                memoryReadings.add(memUsage);
            }

            String batteryUsage = readBatteryUsage();
            batteryReadings.add(batteryUsage);

            try {
                Thread.sleep(sleepDuration);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            if (System.currentTimeMillis() - startTime >= duration) {
                result = new ProfilerResult(batteryReadings, memoryReadings, device, useAragorn);
                break;
            }
        }
        return result;
    }
}