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

package org.tensorflow.lite.examples.imagesegmentation.fragments

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import androidx.recyclerview.widget.GridLayoutManager
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.examples.imagesegmentation.ImageSegmentationHelper
import org.tensorflow.lite.examples.imagesegmentation.ImageSegmentationHelper.SegmentationListener
import org.tensorflow.lite.examples.imagesegmentation.OverlayView
import org.tensorflow.lite.examples.imagesegmentation.OverlayView.ColorLabel
import org.tensorflow.lite.examples.imagesegmentation.R
import org.tensorflow.lite.examples.imagesegmentation.databinding.FragmentCameraBinding
import org.tensorflow.lite.task.vision.segmenter.Segmentation

class CameraFragment : Fragment(), SegmentationListener {

  companion object {
    private const val TAG = "Image Segmentation"
  }

  private var _fragmentCameraBinding: FragmentCameraBinding? = null

  private val fragmentCameraBinding
    get() = _fragmentCameraBinding!!

  private lateinit var imageSegmentationHelper: ImageSegmentationHelper
  private lateinit var bitmapBuffer: Bitmap
  private var preview: Preview? = null
  private var imageAnalyzer: ImageAnalysis? = null
  private var camera: Camera? = null
  private var cameraProvider: ProcessCameraProvider? = null
  private val labelsAdapter: ColorLabelsAdapter by lazy { ColorLabelsAdapter() }

  /** Blocking camera operations are performed using this executor */
  private lateinit var cameraExecutor: ExecutorService

  override fun onResume() {
    super.onResume()
    // Make sure that all permissions are still present, since the
    // user could have removed them while the app was in paused state.
    if (!PermissionsFragment.hasPermissions(requireContext())) {
      Navigation.findNavController(requireActivity(), R.id.fragment_container)
        .navigate(CameraFragmentDirections.actionCameraToPermissions())
    }
  }

  override fun onDestroyView() {
    _fragmentCameraBinding = null
    super.onDestroyView()

    // Shut down our background executor
    cameraExecutor.shutdown()
  }

  override fun onCreateView(
    inflater: LayoutInflater,
    container: ViewGroup?,
    savedInstanceState: Bundle?,
  ): View {
    _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)

    return fragmentCameraBinding.root
  }

  @SuppressLint("MissingPermission")
  override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
    super.onViewCreated(view, savedInstanceState)

    imageSegmentationHelper =
      ImageSegmentationHelper(context = requireContext(), imageSegmentationListener = this)

    // Initialize our background executor
    cameraExecutor = Executors.newSingleThreadExecutor()

    // Wait for the views to be properly laid out
    fragmentCameraBinding.viewFinder.post {
      // Set up the camera and its use cases
      setUpCamera()
    }

    // Attach listeners to UI control widgets
    initBottomSheetControls()

    with(fragmentCameraBinding.recyclerviewResults) {
      adapter = labelsAdapter
      layoutManager = GridLayoutManager(requireContext(), 3)
    }

    fragmentCameraBinding.overlay.setOnOverlayViewListener(
      object : OverlayView.OverlayViewListener {
        override fun onLabels(colorLabels: List<ColorLabel>) {
          // update label at here
          labelsAdapter.updateResultLabels(colorLabels)
        }
      }
    )
  }

  private fun initBottomSheetControls() {
    // When clicked, decrease the number of threads used for segmentation
    fragmentCameraBinding.bottomSheetLayout.threadsMinus.setOnClickListener {
      if (imageSegmentationHelper.numThreads > 1) {
        imageSegmentationHelper.numThreads--
        updateControlsUi()
      }
    }

    // When clicked, increase the number of threads used for segmentation
    fragmentCameraBinding.bottomSheetLayout.threadsPlus.setOnClickListener {
      if (imageSegmentationHelper.numThreads < 4) {
        imageSegmentationHelper.numThreads++
        updateControlsUi()
      }
    }

    // When clicked, change the underlying hardware used for inference. Current options are CPU
    // GPU, and NNAPI
    fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(0, false)
    fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
      object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
          imageSegmentationHelper.currentDelegate = position
          updateControlsUi()
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
          /* no op */
        }
      }
  }

  // Update the values displayed in the bottom sheet. Reset segmenter.
  private fun updateControlsUi() {
    fragmentCameraBinding.bottomSheetLayout.threadsValue.text =
      imageSegmentationHelper.numThreads.toString()

    // Needs to be cleared instead of reinitialized because the GPU
    // delegate needs to be initialized on the thread using it when applicable
    fragmentCameraBinding.overlay.clear()
  }

  // Initialize CameraX, and prepare to bind the camera use cases
  private fun setUpCamera() {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
    cameraProviderFuture.addListener(
      {
        // CameraProvider
        cameraProvider = cameraProviderFuture.get()

        // Build and bind the camera use cases
        bindCameraUseCases()
      },
      ContextCompat.getMainExecutor(requireContext()),
    )
  }

  // Declare and bind preview, capture and analysis use cases
  @SuppressLint("UnsafeOptInUsageError")
  private fun bindCameraUseCases() {

    // CameraProvider
    val cameraProvider =
      cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

    // CameraSelector - makes assumption that we're only using the back camera
    val cameraSelector =
      CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

    // Preview. Only using the 4:3 ratio because this is the closest to our models
    preview =
      Preview.Builder()
        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
        .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
        .build()

    // ImageAnalysis. Using RGBA 8888 to match how our models work
    imageAnalyzer =
      ImageAnalysis.Builder()
        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
        .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .build()
        // The analyzer can then be assigned to the instance
        .also {
          it.setAnalyzer(cameraExecutor) { image ->
            if (!::bitmapBuffer.isInitialized) {
              // The image rotation and RGB image buffer are initialized only once
              // the analyzer has started running
              bitmapBuffer = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            }

            segmentImage(image)
          }
        }

    // Must unbind the use-cases before rebinding them
    cameraProvider.unbindAll()

    try {
      // A variable number of use-cases can be passed here -
      // camera provides access to CameraControl & CameraInfo
      camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

      // Attach the viewfinder's surface provider to preview use case
      preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
    } catch (exc: Exception) {
      Log.e(TAG, "Use case binding failed", exc)
    }
  }

  private fun segmentImage(image: ImageProxy) {
    // Copy out RGB bits to the shared bitmap buffer
    image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

    val imageRotation = image.imageInfo.rotationDegrees
    // Pass Bitmap and rotation to the image segmentation helper for processing and segmentation
    imageSegmentationHelper.segment(bitmapBuffer, imageRotation)
  }

  override fun onConfigurationChanged(newConfig: Configuration) {
    super.onConfigurationChanged(newConfig)
    imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
  }

  // Update UI after objects have been segment. Extracts original image height/width
  // to scale and place bounding boxes properly through OverlayView
  override fun onResults(
    results: List<Segmentation>?,
    inferenceTime: Long,
    imageHeight: Int,
    imageWidth: Int,
  ) {
    activity?.runOnUiThread {
      fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
        String.format("%d ms", inferenceTime)

      // Pass necessary information to OverlayView for drawing on the canvas
      fragmentCameraBinding.overlay.setResults(results, imageHeight, imageWidth)

      // Force a redraw
      fragmentCameraBinding.overlay.invalidate()
    }
  }

  override fun onError(error: String) {
    activity?.runOnUiThread { Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show() }
  }
}
