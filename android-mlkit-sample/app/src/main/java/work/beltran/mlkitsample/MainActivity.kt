package work.beltran.mlkitsample

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

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Input is a 1x1 Tensor
        val inputDims = intArrayOf(1, 1)
        // Output is a 1x1 Tensor
        val outputDims = intArrayOf(1, 1)

        // Define the Input and Output dimensions and types
        val dataOptions = FirebaseModelInputOutputOptions.Builder()
                .setInputFormat(0, FirebaseModelDataType.FLOAT32, inputDims)
                .setOutputFormat(0, FirebaseModelDataType.FLOAT32, outputDims)
                .build()

        // Load mode from local assets storage
        val localModelSource = FirebaseLocalModelSource.Builder("asset")
                .setAssetFilePath("mat_mul.tflite").build()

        val manager = FirebaseModelManager.getInstance()
        manager.registerLocalModelSource(localModelSource)

        val options = FirebaseModelOptions.Builder()
                .setLocalModelName("asset")
                .build()

        val interpreter = FirebaseModelInterpreter.getInstance(options) ?: error("Null Interpreter")

        // Pass the input data, as an array, that contains a single float array, with a single value
        // So in all [0][0] = 21, a 1x1 matrix in the form a Array of FloatArray
        val inp = arrayOf(floatArrayOf(21f))
        val inputs = FirebaseModelInputs.Builder().add(inp).build()

        // Run the model
        interpreter.run(inputs, dataOptions)
                .continueWith { task ->
                    try {
                        // The output is also a Array of FloatArray
                        val output = task.result.getOutput<Array<FloatArray>>(0)

                        // Result is in [0][0]
                        Log.d("MainActivity", "Multiplication result: ${output[0][0]}")
                    } catch (t: Throwable) {
                        Log.e("MainActivity", t.localizedMessage, t)
                    }
                }
    }
}
