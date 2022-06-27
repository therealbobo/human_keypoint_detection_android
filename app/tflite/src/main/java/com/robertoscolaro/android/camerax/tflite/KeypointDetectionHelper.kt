package com.robertoscolaro.android.camerax.tflite

import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.jetbrains.kotlinx.multik.ndarray.data.*


/**
 * Helper class used to communicate between our app and the TF object detection model
 */
class KeypointDetectionHelper(private val tflite: Interpreter) {

    /** Abstraction object that wraps a prediction output in an easy to parse way */
    data class KeypointPrediction(val location: D4Array<Float>)

    private val heatmap = Array(1) {Array(224){Array(224){ FloatArray(FEATURES) } } }

    private val outputBuffer = mapOf(
        0 to heatmap,
    )

    fun predict(image: TensorImage): Bitmap{
        tflite.runForMultipleInputsOutputs(arrayOf(image.buffer), outputBuffer)

        val result = Bitmap.createBitmap(100, 100, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.BLUE


        /*
        // keypoints with mean
        paint.alpha = 100;
        var normArray = FloatArray(FEATURES) { 0F }

        // compute normalizing factor
        for (feature in 0 until FEATURES-1)
            for (i in 0..223)
                for (j in 0..223)
                    normArray[feature] += heatmap[0][i][j][feature]


        var xprob = Array(FEATURES){ FloatArray(224) { 0F } }
        var yprob = Array(FEATURES){ FloatArray(224) { 0F } }

        for (feature in 0 until FEATURES-1)
            for (i in 0..223)
                for (j in 0..223)
                    xprob[feature][i] += (heatmap[0][i][j][feature] / normArray[feature])

        for (feature in 0 until FEATURES-1)
            for (j in 0..223)
                for (i in 0..223)
                    yprob[feature][j] += (heatmap[0][i][j][feature]/normArray[feature])

        var xmean = FloatArray(FEATURES) { 0F }
        var ymean = FloatArray(FEATURES) { 0F }

        for (feature in 0 until FEATURES-1) {
            for (i in 0..223) {
                xmean[feature] += xprob[feature][i] * i
                ymean[feature] += yprob[feature][i] * i
                //Log.e(TAG, String.format("vector: %f",xprob[feature][i] * i))
            }
            //Log.e(TAG, String.format("xmean: %f", xmean[feature]))
            //Log.e(TAG, String.format("ymean: %f", ymean[feature]))
        }

        for (feature in 0 until FEATURES-1) {
            canvas.drawCircle(
                xmean[feature]/224 * 100,
                ymean[feature]/224 * 133,
                1F, paint
            )
            //Log.e(TAG,String.format("feature: %d, x: %f, y: %f",feature,xmean[feature],ymean[feature]))
        }

        */


        // heatmap
        paint.alpha = 50;
        for (feature in 0 until FEATURES-1)
            for (i in 0..223)
                for (j in 0..223)
                    if (heatmap[0][i][j][feature] > THRESHOLD)
                        canvas.drawCircle(
                            j/224F*100,
                            i/224F*133,
                            0.5F, paint)

        return result
    }

    companion object {
        private val TAG = CameraActivity::class.java.simpleName
        const val FEATURES = 17
        const val THRESHOLD = 0.2

    }
}