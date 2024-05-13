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

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.graphics.get
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ColorSpaceType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ImageProperties
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.task.vision.segmenter.ColoredLabel
import org.tensorflow.lite.task.vision.segmenter.OutputType
import org.tensorflow.lite.task.vision.segmenter.Segmentation
import java.lang.Exception
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

/**
 * Class responsible to run the Image Segmentation model. More information about the DeepLab model
 * being used can be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
 * 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
 * 'sofa', 'train', 'tv'
 */
class ImageSegmentationHelper(
    var numThreads: Int = 2,
    var currentDelegate: Int = 0,
    val context: Context,
    val imageSegmentationListener: SegmentationListener?
) {
    private val interpreter: Interpreter = initInterpreter(MODEL_DEEPLABV3)
    private val coloredLabels: List<ColoredLabel> = generateColoredLables()

    @RequiresApi(Build.VERSION_CODES.Q)
    fun segment(image: Bitmap, imageRotation: Int) {

        val (_, H, W, _) = interpreter.getInputTensor(0).shape()

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor =
            ImageProcessor.Builder().add(ResizeOp(H, W, ResizeOp.ResizeMethod.BILINEAR))
                .add(Rot90Op(-imageRotation / 90)).add(NormalizeOp(127.5f, 127.5f)).build()

        // Preprocess the image and convert it into a TensorImage for segmentation.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        // Run Tflite segmentation.
        val segmentResult = segmentWithTflite(tensorImage, interpreter)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        imageSegmentationListener?.onResults(
            segmentResult, inferenceTime, tensorImage.height, tensorImage.width
        )
    }

    fun initInterpreter(filePath: String): Interpreter {
        val assetManager: AssetManager = context.assets
        val bufferModel = loadModelFile(assetManager, filePath)

        val options = Interpreter.Options()
        options.setNumThreads(4)
        options.setUseXNNPACK(true)
        return Interpreter(bufferModel, options)
    }

    data class InferenceData(
        val width: Int,
        val height: Int,
        val channels: Int,
        val buffer: FloatBuffer,
    )

    fun runInference(interpreter: InterpreterApi, preprocessBuffer: ByteBuffer): InferenceData {
        val (_, H, W, C) = interpreter.getOutputTensor(0).shape()
        val result = FloatBuffer.allocate(H * W * C)
        interpreter.run(preprocessBuffer, result)

        return InferenceData(W, H, C, result)
    }

    fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun postprocessImage(inferenceData: InferenceData): ByteBuffer {
        val mask = ByteBuffer.allocateDirect(inferenceData.width * inferenceData.height)
        for (i in 0 until inferenceData.height) {
            for (j in 0 until inferenceData.width) {
                val offset = inferenceData.channels * (i * inferenceData.width + j)

                var maxIndex = 0
                var maxValue = inferenceData.buffer.get(offset)

                for (index in 1 until inferenceData.channels) {
                    if (inferenceData.buffer.get(offset + index) > maxValue) {
                        maxValue = inferenceData.buffer.get(offset + index)
                        maxIndex = index
                    }
                }

                mask.put(i * inferenceData.width + j, maxIndex.toByte())
            }
        }

        return mask
    }

    fun generateColoredLables(): List<ColoredLabel> {
        val labels = listOf<String>(
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv",
            "------"
        )
        val colors = MutableList<ColoredLabelTflite>(labels.size) {
            ColoredLabelTflite(
                labels[0], "", Color.BLACK
            )
        }

        val random = Random()
        val goldenRatioConjugate = 0.618033988749895
        var hue = random.nextDouble()

        // Skip the first label as it's already assigned black
        for (idx in 1 until labels.size) {
            hue += goldenRatioConjugate
            hue %= 1.0
            // Adjust saturation & lightness as needed
            val color = Color.HSVToColor(floatArrayOf(hue.toFloat() * 360, 0.7f, 0.8f))
            colors[idx] = ColoredLabelTflite(labels[idx], "", color)
        }

        return colors
    }

    private fun segmentWithTflite(
        tensorImage: TensorImage, interpreter: Interpreter
    ): List<Segmentation>? {

        // Run inference.
        val inferenceData = runInference(interpreter, tensorImage.tensorBuffer.buffer)

        // Postprocess inference image.
        val mask = postprocessImage(inferenceData)

        // Pack it to TensorImage.
        val imageProp =
            ImageProperties.builder().setWidth(inferenceData.width).setHeight(inferenceData.height)
                .setColorSpaceType(ColorSpaceType.GRAYSCALE).build()
        val maskImage = TensorImage()
        maskImage.load(mask, imageProp)

        val segment = SegmentationTflite(
            OutputType.CATEGORY_MASK, Arrays.asList<TensorImage>(maskImage), coloredLabels
        )

        return Arrays.asList<Segmentation>(segment)
    }

    internal class SegmentationTflite(
        outputType: OutputType?, masks: List<TensorImage>?, coloredLabels: List<ColoredLabel>?
    ) : Segmentation() {
        private var outputType: OutputType? = null
        private var masks: List<TensorImage>? = null
        private var coloredLabels: List<ColoredLabel>? = null

        init {
            if (outputType == null) {
                throw NullPointerException("Null outputType")
            } else {
                this.outputType = outputType
                if (masks == null) {
                    throw NullPointerException("Null masks")
                } else {
                    this.masks = masks
                    if (coloredLabels == null) {
                        throw NullPointerException("Null coloredLabels")
                    } else {
                        this.coloredLabels = coloredLabels
                    }
                }
            }
        }

        override fun getOutputType(): OutputType {
            return outputType!!
        }

        override fun getMasks(): List<TensorImage> {
            return masks!!
        }

        override fun getColoredLabels(): List<ColoredLabel> {
            return coloredLabels!!
        }

        override fun toString(): String {
            return "Segmentation{outputType=" + outputType + ", masks=" + masks + ", coloredLabels=" + coloredLabels + "}"
        }
    }

    internal class ColoredLabelTflite(label: String?, displayName: String?, argb: Int) :
        ColoredLabel() {
        private var label: String? = null
        private var displayName: String? = null
        private var argb = 0

        init {
            if (label == null) {
                throw java.lang.NullPointerException("Null label")
            } else {
                this.label = label
                if (displayName == null) {
                    throw java.lang.NullPointerException("Null displayName")
                } else {
                    this.displayName = displayName
                    this.argb = argb
                }
            }
        }

        override fun getlabel(): String {
            return label!!
        }

        override fun getDisplayName(): String {
            return displayName!!
        }

        override fun getArgb(): Int {
            return argb
        }

        override fun toString(): String {
            return "ColoredLabel{label=" + label + ", displayName=" + displayName + ", argb=" + argb + "}"
        }
    }

    interface SegmentationListener {
        fun onError(error: String)
        fun onResults(
            results: List<Segmentation>?, inferenceTime: Long, imageHeight: Int, imageWidth: Int
        )
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_DEEPLABV3 = "deeplabv3.tflite"

        private const val TAG = "Image Segmentation Helper"
    }
}
