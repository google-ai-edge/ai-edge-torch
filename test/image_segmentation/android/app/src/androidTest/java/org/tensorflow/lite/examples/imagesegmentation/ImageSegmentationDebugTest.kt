// Copyright 2024 The AI Edge Torch Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
//

package org.tensorflow.lite.examples.imagesegmentation

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Environment
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.common.truth.Truth.assertThat
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ColorSpaceType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ImageSegmentationDebugTest {
  companion object {
    private const val TAG = "TFL-SEG"
    private const val PIXEL_SIZE: Int = 3
    private const val INPUT_IMAGE = "input_image.jpg"
    private const val NORMALIZED_IMAGE = "normalized_image.jpg"
    private const val CHANNELS_FIRST_IMAGE = "channels_first_image.jpg"
    private const val SEGMENTATION_MASK = "segmentation_mask.jpg"
    private const val PT_MODEL_FILE = "isnet-general-use.tflite"
    private const val OUTPUT_TENSOR_INDEX: Int = 6
    private val STORAGE_FOLDER =
      File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), "testdata")
  }

  private val interpreter: Interpreter = initInterpreter(PT_MODEL_FILE)

  data class InferenceData(
    val width: Int,
    val height: Int,
    val channels: Int,
    val buffer: FloatBuffer,
  )

  // Main test function that loads image from ASSETS and saves output on STORAGE_FOLDER.
  @Test
  fun executeResultShouldNotChange() {
    // Run segmentation on loaded image.
    loadImage(INPUT_IMAGE)?.let {
      // Run segmentation and save output image on STORAGE_FOLDER.
      segmentWithTflite(it, 0)
      // Check if the image exists.
      val status = File(STORAGE_FOLDER, SEGMENTATION_MASK).exists()
      assertThat(status).isTrue()
    }
  }

  // Run segmentation on loaded image.
  private fun segmentWithTflite(image: Bitmap, imageRotation: Int) {
    val (_, C, H, W) = interpreter.getInputTensor(0).shape()

    // Preprocess the image and convert it into a TensorImage for segmentation.
    val imageProcessor =
      ImageProcessor.Builder()
        .add(ResizeOp(H, W, ResizeOp.ResizeMethod.BILINEAR))
        .add(Rot90Op(-imageRotation / 90))
        .add(NormalizeOp(0.0f, 255.0f))
        .build()
    val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

    // Save normalized image for debug purposes..
    saveBitmapOnStorage(tensorImage.tensorBuffer, NORMALIZED_IMAGE)

    // Change to channels first (CHW) layout.
    val tensorChannelsFirstImage = makeChannelsFirst(tensorImage)

    // Save channels first image for debug purposes..
    saveBitmapOnStorage(tensorChannelsFirstImage, CHANNELS_FIRST_IMAGE)

    // Inference time is the difference between the system time at the start and finish of the
    // process
    var inferenceTime = SystemClock.uptimeMillis()

    // Run inference.
    val inferenceData = runInference(interpreter, tensorChannelsFirstImage.buffer)

    inferenceTime = SystemClock.uptimeMillis() - inferenceTime
    Log.i(TAG, ">> TFLite inference time (ms): " + inferenceTime)

    // Post-process inference resut and create segmentation mask.
    val maskImage = doPostProcessing(inferenceData)

    // Save segmentation mask on STORAGE_FOLDER.
    saveBitmapOnStorage(maskImage, SEGMENTATION_MASK)

    return
  }

  // Load *.tflite model from ASSETS.
  private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
    Log.i(TAG, ">> Loading file from ASSETS: " + modelPath)
    val fileDescriptor = assetManager.openFd(modelPath)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
  }

  // Initialize interpreter to run inference on CPU with 4 threads on XNNPACK.
  private fun initInterpreter(filePath: String): Interpreter {
    Log.i(TAG, ">> Initializing interpreter with: " + filePath)
    val assetManager: AssetManager = InstrumentationRegistry.getInstrumentation().context.assets
    val bufferModel = loadModelFile(assetManager, filePath)

    val options = Interpreter.Options()
    options.setNumThreads(4)
    options.setUseXNNPACK(true)
    return Interpreter(bufferModel, options)
  }

  @Throws(Exception::class)
  private fun loadImage(fileName: String): Bitmap? {
    Log.i(TAG, ">> Loading image from ASSETS: " + fileName)
    val assetManager: AssetManager = InstrumentationRegistry.getInstrumentation().context.assets
    val inputStream: InputStream = assetManager.open(fileName)
    return BitmapFactory.decodeStream(inputStream)
  }

  // Change layout to meet Pytorch channels first requirement.
  private fun makeChannelsFirst(image: TensorImage): TensorBuffer {
    Log.i(TAG, ">> Changing layout to channels first...")
    val inArray: FloatArray = image.tensorBuffer.floatArray
    val outArray: FloatArray = FloatArray(inArray.size)
    val stride = image.height * image.width
    for (i in 0 until image.height * image.width) {
      val r = inArray[PIXEL_SIZE * i]
      val g = inArray[PIXEL_SIZE * i + 1]
      val b = inArray[PIXEL_SIZE * i + 2]
      outArray[i] = r
      outArray[stride + i] = g
      outArray[2 * stride + i] = b
    }

    val channelsFirstImage =
      TensorBuffer.createFrom(image.tensorBuffer, image.tensorBuffer.dataType)
    channelsFirstImage.loadArray(outArray)

    return channelsFirstImage
  }

  // Run multiple inputs and multiple outputs inference.
  private fun runInference(
    interpreter: InterpreterApi,
    preprocessBuffer: ByteBuffer,
  ): InferenceData {
    var W = 0
    var H = 0
    var C = 0

    val multipleInputs = arrayOf<Any>(preprocessBuffer)
    val multipleOutputs: MutableMap<Int, Any> = HashMap()
    val floatBufferArray = Array<FloatBuffer?>(interpreter.outputTensorCount) { null }
    for (i in 0 until interpreter.outputTensorCount) {
      val (_, c, h, w) = interpreter.getOutputTensor(i).shape()
      floatBufferArray[i] = FloatBuffer.allocate(h * w * c)
      multipleOutputs[i] = floatBufferArray[i]!!

      if (i == OUTPUT_TENSOR_INDEX) {
        W = w
        H = h
        C = c
      }
    }
    Log.i(TAG, ">> Running inference...")
    interpreter.runForMultipleInputsOutputs(multipleInputs, multipleOutputs)

    return InferenceData(W, H, C, floatBufferArray[OUTPUT_TENSOR_INDEX]!!)
  }

  // Do inference post-processing and create an segementation mask.
  private fun doPostProcessing(inferenceData: InferenceData): TensorBuffer {
    Log.i(TAG, ">> Post-processing...")

    val outArray = inferenceData.buffer.array()
    val maxVal = outArray.maxOrNull()!!
    val minVal = outArray.minOrNull()!!
    outArray.map { (it - minVal) / (maxVal - minVal) }

    // Create RGB segmentation mask.
    val shape = intArrayOf(inferenceData.width, inferenceData.height)
    val tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.FLOAT32)
    tensorBuffer.loadArray(outArray)

    return tensorBuffer
  }

  fun saveBitmapOnStorage(tensorBuffer: TensorBuffer, filename: String) {
    if (!STORAGE_FOLDER.exists()) {
      STORAGE_FOLDER.mkdirs()
    }

    val outArray = ByteArray(PIXEL_SIZE * tensorBuffer.shape[0] * tensorBuffer.shape[1])
    for (i in 0 until tensorBuffer.shape[0] * tensorBuffer.shape[1]) {
      // Cast float32 [0..1.0] to uint8 [0..255]
      val pixel =
        if (tensorBuffer.dataType == DataType.FLOAT32) (255 * tensorBuffer.getFloatValue(i))
        else tensorBuffer.getIntValue(i)

      outArray[PIXEL_SIZE * i] = pixel.toByte()
      outArray[PIXEL_SIZE * i + 1] = pixel.toByte()
      outArray[PIXEL_SIZE * i + 2] = pixel.toByte()
    }

    // Create RGB image.
    val shape = intArrayOf(tensorBuffer.shape[0], tensorBuffer.shape[1], PIXEL_SIZE)
    val tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.UINT8)
    val byteBuffer = ByteBuffer.allocate(outArray.size).put(outArray)
    tensorBuffer.loadBuffer(byteBuffer)

    val tensorImage = TensorImage()
    tensorImage.load(tensorBuffer, ColorSpaceType.RGB)

    val file = File(STORAGE_FOLDER, filename)
    FileOutputStream(file).use { out ->
      tensorImage.bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
      Log.e(TAG, ">> Saving bitmap to: " + file.absolutePath)
    }
  }
}
