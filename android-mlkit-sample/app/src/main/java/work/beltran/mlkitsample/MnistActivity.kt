package work.beltran.mlkitsample

import android.graphics.BitmapFactory
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import android.util.Log
import com.google.firebase.ml.custom.FirebaseModelDataType
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions
import com.google.firebase.ml.custom.FirebaseModelInputs
import com.google.firebase.ml.custom.FirebaseModelInterpreter
import com.google.firebase.ml.custom.FirebaseModelManager
import com.google.firebase.ml.custom.FirebaseModelOptions
import com.google.firebase.ml.custom.model.FirebaseLocalModelSource

class MnistActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_main)

        // Input is a 1x1 Tensor
        val inputDims = intArrayOf(1, 784)
        // Output is a 1x1 Tensor
        val outputDims = intArrayOf(1, 10)

        // Define the Input and Output dimensions and types
        val dataOptions = FirebaseModelInputOutputOptions.Builder()
                .setInputFormat(0, FirebaseModelDataType.FLOAT32, inputDims)
                .setOutputFormat(0, FirebaseModelDataType.FLOAT32, outputDims)
                .build()

        // Load mode from local assets storage
        val localModelSource = FirebaseLocalModelSource.Builder("asset")
                .setAssetFilePath("nmist_mlp.tflite").build()

        val manager = FirebaseModelManager.getInstance()
        manager.registerLocalModelSource(localModelSource)

        val options = FirebaseModelOptions.Builder()
                .setLocalModelName("asset")
                .build()

        val interpreter = FirebaseModelInterpreter.getInstance(options) ?: error("Null Interpreter")

        // Just passing zeroes (it's a gray picture)
        val inp = arrayOf(FloatArray(784) { 0f })


        val bitmap = BitmapFactory.decodeStream(assets.open("three.bmp"))

        Log.d("MainActivity", "${bitmap.width} x ${bitmap.height}")

        val intValues = IntArray(784)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        intValues.forEachIndexed { index, i ->
            val f = (i and 0xFF).toFloat()
            inp[0][index] = f / 256
        }

        val inputs = FirebaseModelInputs.Builder().add(inp).build()

        // Run the model
        interpreter.run(inputs, dataOptions)
                .addOnFailureListener {
                    Log.e("MainActivity", it.localizedMessage, it)
                }
                .continueWith { task ->
                    try {
                        // The output is also a Array of FloatArray
                        val output = task.result.getOutput<Array<FloatArray>>(0)

                        // Result is in [0][x]
                        Log.d("MainActivity", "Result: ${output[0].joinToString()}")
                    } catch (t: Throwable) {
                        Log.e("MainActivity", t.localizedMessage, t)
                    }
                }
    }
}
