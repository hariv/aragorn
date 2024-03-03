package com.gunrock.aragornbasic;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;

import androidx.annotation.Nullable;

public class OverlayView extends View {
    private float left, top, right, bottom;

    public OverlayView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        left = -Float.MAX_VALUE;
        top = -Float.MAX_VALUE;
        right = -Float.MAX_VALUE;
        bottom = -Float.MAX_VALUE;
    }


    public void setBB(float left, float top, float right, float bottom) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
    }

    @Override
    public void draw(Canvas canvas) {
        super.draw(canvas);

        canvas.drawColor(Color.GRAY);
        if (left != -Float.MAX_VALUE && top != -Float.MAX_VALUE && right != -Float.MAX_VALUE && bottom != -Float.MAX_VALUE) {
            Log.d("debug", "inside draw");
            //Log.d("left", String.valueOf(left));
            //Log.d("top", String.valueOf(top));
            //Log.d("right", String.valueOf(right));
            //Log.d("bottom", String.valueOf(bottom));
            //canvas.drawColor(Color.GRAY);

            Paint paint = new Paint();
            paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
            canvas.drawRect(left, top, right, bottom, paint);
            /*Paint paint = new Paint();
            paint.setColor(Color.argb(128, 0, 0, 0));
            canvas.drawRect(0, 0, getWidth(), getHeight(), paint);
            //paint.setColor(Color.TRANSPARENT);

            RectF rect = new RectF(left, top, right, bottom);
            //canvas.drawRect(rect, paint);
            canvas.drawRect(rect, new Paint());*/
        }
    }

}
