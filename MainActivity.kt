package com.example.intrusion_detection

import android.Manifest
import android.content.ContentValues.TAG
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ColorSpace.Model
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.intrusion_detection.ml.LiteModelImagenetMobilenetV3Small075224Classification5Default1
import com.example.intrusion_detection.ml.New
//import com.example.intrusion_detection.ml.New
//import kotlinx.coroutines.flow.internal.NoOpContinuation.context
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.lang.Math.exp
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp



class MainActivity : ComponentActivity() {


    lateinit var textureView: TextureView
    lateinit var imageView: ImageView
    lateinit var textView: TextView
    lateinit var cameraManager: CameraManager
    lateinit var handler: Handler
    lateinit var cameraDevice: CameraDevice
    lateinit var imageProcessor: ImageProcessor
    lateinit var bitmap: Bitmap
    lateinit var byteBuffer: ByteBuffer
    lateinit var button: Button


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        get_permission()
        imageView = findViewById(R.id.imageView)
        textView = findViewById(R.id.textView)
        button = findViewById(R.id.btn)
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)).build()
        val model = New.newInstance(this)
//        val model = LiteModelImagenetMobilenetV3Small075224Classification5Default1.newInstance(this)
//        var labels=application.assets.open(labels.txt).bufferedReader().readLines()
//        var label = application.assets.open(labels).bufferedReader().readLines()
        textureView = findViewById(R.id.textureView)
        textureView.surfaceTextureListener=object:TextureView.SurfaceTextureListener{
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()
            }
            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {

                bitmap = textureView.bitmap!!
                var tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(bitmap)
                tensorImage = imageProcessor.process(tensorImage)


// Creates inputs for reference.
               val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 3, 224, 224), DataType.FLOAT32)
              inputFeature0.loadBuffer(tensorImage.buffer)
            val outputs = model.process(inputFeature0)
            val outputFeature = outputs.outputFeature0AsTensorBuffer.floatArray
//                Log.d("MyApp", "Output feature: ${outputFeature.contentToString()}")
                var mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                var x = 0
                outputFeature.forEachIndexed { index, fl ->
                    x = index
                    x *= 4
                    Log.d(TAG, "onSurfaceTextureUpdated:${index},${fl}")
                    if (index==0){
                        textView.setText("achyut")
                    }
                }
//
//                Log.d("MyApp", "Output feature: ${softmax(outputFeature)}")
//                val model = New.newInstance(this@MainActivity)
//                bitmap = textureView.bitmap!!
//                var image = TensorImage.fromBitmap(bitmap)
//                val inputSize = 224
//                val numBytesPerChannel =4
//                val bufferSize = inputSize * inputSize * numBytesPerChannel * 3 // 3 channels for RGB
//                val inputBuffer = ByteBuffer.allocateDirect(bufferSize)





// Creates inputs for reference.
//                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 3, 224, 224), DataType.FLOAT32)
//                inputFeature0.loadBuffer(inputBuffer)

// Runs model inference and gets result.
//                val outputs = model.process(inputFeature0)
//                val outputFeature = outputs.outputFeature0AsTensorBuffer.floatArray
//                if(outputFeature[0]>outputFeature[1])
//                {
//                    textView.setText("ACHYUT")
//                }
//                else{
//                    textView.setText("ANUJ")
//
//                }


//                val timestamp = System.currentTimeMillis()
////                val outputString = outputFeature.floatArray.joinToString(separator = "\n")
//               textView.text = ("Output feature[0]: ${outputFeature[0]}")
//                Log.d("MyApp", "Output feature[0]: ${outputs}")

                imageView.setImageBitmap(mutable)

            }

        }
        cameraManager = getSystemService(CAMERA_SERVICE)as CameraManager
        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)


        button.setOnClickListener {
            model.close()

        }

    }

    fun softmax(input: FloatArray): FloatArray {
        val output = FloatArray(input.size)
        var sum = 0.0f
        for (i in input.indices) {
            output[i] = exp(input[i])
            sum += output[i]
        }
        for (i in output.indices) {
            output[i] /= sum
        }
        return output
    }

    override fun onDestroy() {
        super.onDestroy()
    }
    fun open_camera() {
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return
        }
        cameraManager.openCamera(cameraManager.cameraIdList[0],object :CameraDevice.StateCallback(){
            override fun onOpened(p0: CameraDevice) {
                cameraDevice = p0
                var surfaceTexture = textureView.surfaceTexture
                var surface = Surface(surfaceTexture)
                var captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                captureRequest.addTarget(surface)

                cameraDevice.createCaptureSession(listOf(surface), object: CameraCaptureSession.StateCallback(){
                    override fun onConfigured(p0: CameraCaptureSession) {
                        p0.setRepeatingRequest(captureRequest.build(), null, null)
                    }
                    override fun onConfigureFailed(p0: CameraCaptureSession) {
                    }
                }, handler)



            }

            override fun onDisconnected(p0: CameraDevice) {
            }

            override fun onError(p0: CameraDevice, p1: Int) {
            }
        }, handler)
    }

    fun get_permission() {
        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if(grantResults[0] != PackageManager.PERMISSION_GRANTED){
            get_permission()
        }
    }



}


