package work.beltran.spamdetektor

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

        val inputDims = intArrayOf(1, 1)
        val outputDims = intArrayOf(1, 1)

        val dataOptions = FirebaseModelInputOutputOptions.Builder()
                .setInputFormat(0, FirebaseModelDataType.FLOAT32, inputDims)
                .setOutputFormat(0, FirebaseModelDataType.FLOAT32, outputDims)
                .build()

        val localModelSource = FirebaseLocalModelSource.Builder("asset")
                .setAssetFilePath("mat_mul.tflite").build()

        val manager = FirebaseModelManager.getInstance()
        manager.registerLocalModelSource(localModelSource)

        val options = FirebaseModelOptions.Builder()
                .setLocalModelName("asset")
                .build()

        val interpreter = FirebaseModelInterpreter.getInstance(options) ?: error("Null Interpreter")

        val inp = arrayOf(floatArrayOf(1f))
//        val inp = floatArrayOf(1f)
        val inputs = FirebaseModelInputs.Builder().add(inp).build()
        interpreter.run(inputs, dataOptions)
                .continueWith { task ->
                    try {
                        val output = task.result.getOutput<Array<FloatArray>>(0)
                        Log.d("MainActivity", output[0][0].toString())
                    } catch (t: Throwable) {
                        Log.e("MainActivity", t.localizedMessage, t)
                    }
                }




    }
}
