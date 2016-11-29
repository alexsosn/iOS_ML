# Machine Learning for iOS 

## Tools and resources to create really smart iOS applications.

**Last Update: November 29, 2016.**

[Link to a blog post](http://alexsosn.github.io/ml/2015/11/05/iOS-ML.html)

Curated list of resources for iOS developers in following topics: 

- Machine Learning, 
- Artificial Intelligence,
- Natural Language Processing (NLP), 
- Computer Vision, 
- General-Purpose GPU Computing (GPGPU), 
- Data Visualization,
- Bioinformatics

Most of the de-facto standard tools in domains listed above are written in iOS-unfriendly languages (Python/Java/R/Matlab) so find something appropriate for your iOS application may be a challenging task.
This list consists mainly of libraries written in Objective-C, Swift, C, C++, JavaScript and some other languages if they can be easily ported to iOS. Also links to some relevant web APIs, blog posts, videos and learning materials included.

**Pull-requests are welcome [here](https://github.com/alexsosn/iOS_ML)!**

## Where to learn about machine learning and related topics in general
* [Courserra course](https://www.coursera.org/learn/machine-learning/home/info) on machine learning from [Andrew Ng](https://twitter.com/AndrewYNg).
* [Machine learning playlist on Youtube](https://www.youtube.com/playlist?list=PLD0F06AA0D2E8FFBA).
* Free online interactive book ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/).
* ["Natural Language Processing with Python"](http://www.nltk.org/book/) - free online book.
* [Heterogeneous Parallel Programming](https://www.coursera.org/course/hetero) course.
* Immersive Linear Algebra [interactive book](http://immersivemath.com/ila/index.html) by J. Ström, K. Åström, and T. Akenine-Möller.
* [Videos](http://videolectures.net/deeplearning2015_montreal/) from Deep Learning Summer School, Montreal 2015.
* [Deep Learning for Perception](https://computing.ece.vt.edu/~f15ece6504/) by Virginia Tech, Electrical and Computer Engineering, Fall 2015: ECE 6504
* [Probabilistic Programming & Bayesian Methods for Hackers](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/) - An intro to Bayesian methods and probabilistic programming from a computation/understanding-first, mathematics-second point of view. 
* [CAP 5415 - Computer Vision](http://crcv.ucf.edu/courses/CAP5415/Fall2014/index.php) by UCF
* [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html) by Stanford
* [Machine Learning: 2014-2015 Course materials](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/) by Oxford
* [Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition.](http://cs231n.stanford.edu/)
* [Deep Learning for Natural Language Processing \(without Magic\)](http://nlp.stanford.edu/courses/NAACL2013/)

## Web APIs

* [**IBM** Watson](http://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/) - Enable Cognitive Computing Features In Your App Using IBM Watson's Language, Vision, Speech and Data APIs.
	* [Introducing the (beta) IBM Watson iOS SDK](https://developer.ibm.com/swift/2015/12/18/introducing-the-new-watson-sdk-for-ios-beta/)
* [AlchemyAPI](http://www.alchemyapi.com/) - Semantic Text Analysis APIs Using Natural Language Processing. Now part of IBM Watson.
* [**Microsoft** Project Oxford](https://www.projectoxford.ai/)
* [**Google** Prediction engine](https://cloud.google.com/prediction/docs)
	* [Objective-C API](https://code.google.com/p/google-api-objectivec-client/wiki/Introduction)
* [Google Translate API](https://cloud.google.com/translate/docs)
* [Google Cloud Vision API](https://cloud.google.com/vision/)
* [**Amazon** Machine Learning](http://aws.amazon.com/documentation/machine-learning/) - Amazon ML is a cloud-based service for developers. It provides visualization tools to create machine learning models. Obtain predictions for application using APIs. 
	* [iOS developer guide](https://docs.aws.amazon.com/mobile/sdkforios/developerguide/getting-started-machine-learning.html).
	* [iOS SDK](https://github.com/aws/aws-sdk-ios)
* [**PredictionIO**](https://prediction.io/) - opensource machine learning server for developers and ML engineers. Built on Apache Spark, HBase and Spray.
	* [Swift SDK](https://github.com/minhtule/PredictionIO-Swift-SDK)
	* [Tapster iOS Demo](https://github.com/minhtule/Tapster-iOS-Demo) - This demo demonstrates how to use the PredictionIO Swift SDK to integrate an iOS app with a PredictionIO engine to make your mobile app more interesting.
	* [Tutorial](https://github.com/minhtule/Tapster-iOS-Demo/blob/master/TUTORIAL.md) on using Swift with PredictionIO.
* [**Wit.AI**](https://wit.ai/) - NLP API
* [**Yandex** SpeechKit](https://tech.yandex.com/speechkit/mobilesdk/) Text-to-speech and speech-to-text for Russian language. iOS SDK available.
* [**Abbyy** OCR SDK](http://www.abbyy.com/mobile-ocr/iphone-ocr/)
* [**Clarifai**](http://www.clarifai.com/#) - deep learning web api for image captioning. [iOS starter project](https://github.com/Clarifai/clarifai-ios-starter)
* [**MetaMind**](https://www.metamind.io/) - deep learning web api for image captioning.
* [Api.AI](https://api.ai/) - Build intelligent speech interfaces
for apps, devices, and web


## General-Purpose Machine Learning Libraries

* [FANN](https://cocoapods.org/pods/FANN) - Fast Artifical Neural Network library; an implementation of neural networks.
* [lbimproved](https://github.com/lemire/lbimproved) - DTW + kNN in C
* **Shark** - provides libraries for the design of adaptive systems, including methods for linear and nonlinear optimization (e.g., evolutionary and gradient-based algorithms), kernel-based algorithms and neural networks, and other machine learning techniques. [CocoaPods](https://cocoapods.org/pods/Shark-SDK). [Official site](http://image.diku.dk/shark/sphinx_pages/build/html/index.html)
* [YCML](http://yconst.com/software/ycml/) - A Machine Learning framework for Objective-C and Swift (OS X / iOS). 
The following algorithms are currently available: Gradient Descent Backpropagation, Resilient Backpropagation (RProp), Extreme Learning Machines (ELM), Forward Selection using Orthogonal Least Squares (for RBF Net), also with the PRESS statistic, Binary Restricted Boltzmann Machines (CD & PCD, Untested!). YCML also contains some optimization algorithms as support for deriving predictive models, although they can be used for any kind of problem: Gradient Descent (Single-Objective, Unconstrained), RProp Gradient Descent (Single-Objective, Unconstrained), NSGA-II (Multi-Objective, Constrained).
	* [Github](https://github.com/yconst/ycml/). 
* [Swix](https://github.com/scottsievert/swix) - Swift implementation of NumPy.
* [Brain](https://github.com/harthur/brain) - Neural networks in JavaScript. Unmaintained.
* [April-ANN](https://github.com/pakozm/april-ann) - APRIL-ANN toolkit (A Pattern Recognizer In Lua with ANNs). This toolkit incorporates ANN algorithms (as dropout, stacked denoising auto-encoders, convolutional NNs), with other pattern recognition methods as HMMs among others. Additionally, in experimental stage, it is possible to perform automatic differentiation, for advanced ML research. Potentially can be ported for iOS. 
	* [Official Site](http://pakozm.github.com/april-ann/). 
* [Recommender](https://github.com/GHamrouni/Recommender) - A C library for product recommendations/suggestions using collaborative filtering (CF).
* [SNNeuralNet](https://github.com/devongovett/SNNeuralNet) - A neural network library for Objective-C based on brain.js, for iOS and Mac OS X.
* [MLPNeuralNet](https://github.com/nikolaypavlov/MLPNeuralNet) - Fast multilayer perceptron neural network library for iOS and Mac OS X. MLPNeuralNet predicts new examples by trained neural network. It is built on top of the Apple's Accelerate Framework, using vectorized operations and hardware acceleration if available.
* [MAChineLearning](https://github.com/gianlucabertani/MAChineLearning) - An Objective-C multilayer perceptron library, with full support for training through backpropagation. Implemented using vDSP and vecLib, it's 20 times faster than its Java equivalent. Includes sample code for use from Swift.
* [BrainCore](https://github.com/aleph7/BrainCore) - simple but fast neural network framework written in Swift. It uses Upsurge and the underlying Accelerate framework to be as fast as possible.
* [Swift-AI](https://github.com/collinhundley/Swift-AI) - 3-layer NN.
* [EERegression](https://github.com/erkekin/EERegression/) -  General purpose multivaritate and quadratic Regression library for Swift 2.1 
* [SwiftSimpleNeuralNetwork](https://github.com/davecom/SwiftSimpleNeuralNetwork) - Feed forward and back propagation.

* **Kalvar Lin's libraries**
	* [ios-BPN-NeuralNetwork](https://github.com/Kalvar/ios-BPN-NeuralNetwork) - 3 layers NN + Back Propagation.
	* [ios-Multi-Perceptron-NeuralNetwork / KRANN](https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork) - Multi-perceptrons neural NN based on Back Propagation NN.
	* [ios-KRHebbian-Algorithm](https://github.com/Kalvar/ios-KRHebbian-Algorithm)
	* [ios-KRKmeans-Algorithm](https://github.com/Kalvar/ios-KRKmeans-Algorithm) - K-Means clustering and classification.
	* [ios-KRFuzzyCMeans-Algorithm](https://github.com/Kalvar/ios-KRFuzzyCMeans-Algorithm) - Fuzzy C-Means fuzzy clustering / classification algorithm.
	* [ios-KRGreyTheory](https://github.com/Kalvar/ios-KRGreyTheory) - Grey Theory implementation \([this one?](http://researchinformation.co.uk/grey/IntroGreySysTheory.pdf)\).
	* [ios-KRKNN](https://github.com/Kalvar/ios-KRKNN) - kNN implementation.
	* [ios-ML-Recommendation-System](https://github.com/Kalvar/ios-ML-Recommendation-System)


### Deep Learning
* [Torch-iOS](https://github.com/clementfarabet/torch-ios) - [Torch](http://torch.ch/) port for iOS. Torch is a scientific computing framework with wide support for machine learning algorithms. One of the most popular deep learning frameworks.
* [Caffe](http://caffe.berkeleyvision.org) - A deep learning framework developed with cleanliness, readability, and speed in mind.
[GitHub](https://github.com/BVLC/caffe). [BSD]
	* [iOS port](https://github.com/aleph7/caffe)
	* C++ examples: [Classifying ImageNet](http://caffe.berkeleyvision.org/gathered/examples/cpp_classification.html), [Extracting Features](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html)
	* [Caffe iOS sample](https://github.com/noradaiko/caffe-ios-sample)
* [Deep Belief SDK](https://github.com/jetpacapp/DeepBeliefSDK) -  The SDK for Jetpac's iOS Deep Belief image recognition framework
* [Convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/) - ConvNetJS is a Javascript library for training Deep Learning models by [Andrej Karpathy](https://twitter.com/karpathy). [GitHub](https://github.com/karpathy/convnetjs)
	* [ConvNetSwift](https://github.com/alexsosn/ConvNetSwift) - Swift port [work in progress].
* [MXNet](http://mxnet.readthedocs.org/en/latest/) - MXNet is a deep learning framework designed for both efficiency and flexibility.
	* [Deploying mxnet to smartphone](https://github.com/dmlc/mxnet/tree/master/amalgamation)
* [TensorFlow](http://www.tensorflow.org/) - an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. TensorFlow Will have iOS support in the future.
* [DeepLearningKit](http://deeplearningkit.org/) - Open Source Deep Learning Framework from Memkite for Apple's tvOS, iOS and OS X.
* [BNNS](https://developer.apple.com/reference/accelerate/1912851-bnns) - Apple Basic neural network subroutines (BNNS) is a collection of functions that you use to implement and run neural networks, using previously obtained training data.
	* [BNNS usage examples](https://github.com/shu223/iOS-10-Sampler) in iOS 10 sampler.


## AI
* [Mendel](https://github.com/saniul/Mendel) - Genetic algorithms in Swift.

### Game AI
* [Introduction to AI Programming for Games](http://www.raywenderlich.com/24824/introduction-to-ai-programming-for-games)
* [dlib](http://dlib.net/) is a library which has many useful tools including machine learning.
* [MicroPather](http://www.grinninglizard.com/MicroPather/) is a path finder and A* solver (astar or a-star) written in platform independent C++ that can be easily integrated into existing code.
* Here is a [list](http://www.ogre3d.org/tikiwiki/List+Of+Libraries#Artificial_intelligence) of some AI libraries suggested on OGRE3D website. Seems they are mostly written in C++.
* [GameplayKit Programming Guide](https://developer.apple.com/library/content/documentation/General/Conceptual/GameplayKit_Guide/)


## Natural Language Processing
* [Parsimmon](https://github.com/ayanonagon/Parsimmon)
* [NSLinguisticTagger](http://nshipster.com/nslinguistictagger/)

### Computational Semantics
* [Word2Vec](https://code.google.com/p/word2vec/) - Original C implementation of Word2Vec Deep Learning algorithm. Works on iPhone like a charm.

### Text Mining
* [Twitter text](https://github.com/twitter/twitter-text-objc) - 
An Objective-C implementation of Twitter's text processing library. The library includes methods for extracting user names, mentions headers, hashtags, and more – all the tweet specific language syntax you could ever want.

### Speech Recognition (TTS) and Generation (STT)
* [TLSphinx](https://github.com/tryolabs/TLSphinx), [Tutorial](http://blog.tryolabs.com/2015/06/15/tlsphinx-automatic-speech-recognition-asr-in-swift/)
* [MVSpeechSynthesizer](https://github.com/vimalmurugan89/MVSpeechSynthesizer)
* [OpenEars™: free speech recognition and speech synthesis for the iPhone](http://www.politepix.com/openears/) - OpenEars™ makes it simple for you to add offline speech recognition and synthesized speech/TTS to your iPhone app quickly and easily. It lets everyone get the great results of using advanced speech UI concepts like statistical language models and finite state grammars in their app, but with no more effort than creating an NSArray or NSDictionary. 
	* [Tutorial (Russian)](http://habrahabr.ru/post/237589/)
* [Kaldi-iOS framework](http://keenresearch.com/) - on-device speech recognition using deep learning.
	* [Proof of concept app](https://github.com/keenresearch/kaldi-ios-poc)


## Computer Vision
* [**Vuforia** Framework](https://developer.vuforia.com/downloads/sdk)
	* [Object detection with Vuforia tutorial.](http://habrahabr.ru/company/dataart/blog/256385/) \(Russian\)

* **[OpenCV](http://opencv.org)** - Open Source Computer Vision Library. [BSD]
	* [OpenCV crash course](http://www.pyimagesearch.com/free-opencv-crash-course/
) 
	* [OpenCVSwiftStitch](https://github.com/foundry/OpenCVSwiftStitch)
	* [Tutorial: using and building openCV on iOS devices](http://maniacdev.com/2011/07/tutorial-using-and-building-opencv-open-computer-vision-on-ios-devices)

* [trackingjs](http://trackingjs.com/) - Object tracking in JS

### Text Recognition (OCR)
* **Tesseract**
	* [Install and Use Tesseract on iOS](http://lois.di-qual.net/blog/install-and-use-tesseract-on-ios-with-tesseract-ios/)
	* [tesseract-ios-lib](https://github.com/ldiqual/tesseract-ios-lib)
	* [tesseract-ios](https://github.com/ldiqual/tesseract-ios)
	* [Tesseract-OCR-iOS](https://github.com/gali8/Tesseract-OCR-iOS)
	* [OCR-iOS-Example](https://github.com/robmathews/OCR-iOS-Example)
* [ocrad.js](https://github.com/antimatter15/ocrad.js) - JS OCR

## General Math
* [Surge](https://github.com/mattt/Surge) from Mattt
* [YCMatrix](https://github.com/yconst/YCMatrix) - A flexible Matrix library for Objective-C and Swift (OS X / iOS)
* [Eigen](http://eigen.tuxfamily.org/) - A high-level C++ library of template headers for linear algebra, matrix and vector operations, numerical solvers and related algorithms. [MPL2]

## GPGPU

### Articles
* [OpenCL for iOS](https://github.com/linusyang/opencl-test-ios) - just a test.
* Exploring GPGPU on iOS. 
	* [Article](http://ciechanowski.me/blog/2014/01/05/exploring_gpgpu_on_ios/) 
	* [Code](https://github.com/Ciechan/Exploring-GPGPU-on-iOS
)

* GPU-accelerated video processing for Mac and iOS. [Article](http://www.sunsetlakesoftware.com/2010/10/22/gpu-accelerated-video-processing-mac-and-ios0).

* [Concurrency and OpenGL ES](https://developer.apple.com/library/ios/documentation/3ddrawing/conceptual/opengles_programmingguide/ConcurrencyandOpenGLES/ConcurrencyandOpenGLES.html) - Apple programming guide.

* [OpenCV on iOS GPU usage](http://stackoverflow.com/questions/10704916/opencv-on-ios-gpu-usage) - SO discussion.

#### Metal
* Simon's Gladman \(aka flexmonkey\) [blog](http://flexmonkey.blogspot.com/)
	* [Talk on iOS GPU programming](https://realm.io/news/altconf-simon-gladman-ios-gpu-programming-with-swift-metal/) with Swift and Metal at Realm Altconf.
	* [The Supercomputer In Your Pocket:
Metal & Swift](https://realm.io/news/swift-summit-simon-gladman-metal/) - a video from the Swift Summit Conference 2015
	* https://github.com/FlexMonkey/MetalReactionDiffusion
	* https://github.com/FlexMonkey/ParticleLab
* [Memkite blog](http://memkite.com/) - startup intended to create deep learning library for iOS.
	* [Swift and Metal example for General Purpose GPU Processing on Apple TVOS 9.0](https://github.com/memkite/MetalForTVOS)
	* [Data Parallel Processing with Swift and Metal on GPU for iOS8](https://github.com/memkite/SwiftMetalGPUParallelProcessing)
	* [Example of Sharing Memory between GPU and CPU with Swift and Metal for iOS8](http://memkite.com/blog/2014/12/30/example-of-sharing-memory-between-gpu-and-cpu-with-swift-and-metal-for-ios8/)
* [Metal by Example blog](http://metalbyexample.com/)
* [objc-io article on Metal](https://www.objc.io/issues/18-games/metal/)

### GPU-accelerated libraries
* [GPUImage](https://github.com/BradLarson/GPUImage) is a GPU-accelerated image processing library.

## Data Visualization
* [iOS-Charts](https://github.com/danielgindi/ios-charts)
* [Core Plot](https://github.com/core-plot/core-plot)
* [Awesome iOS charts](https://github.com/sxyx2008/awesome-ios-chart)
* [JTChartView](https://github.com/kubatru/JTChartView)
* [VTK](http://www.vtk.org/gallery/)
	* [VTK in action](http://www.vtk.org/vtk-in-action/)

## Bioinformatics
* [BioJS](http://biojs.net/) - a set of tools for bioinformatics in the browser. BioJS builds a infrastructure, guidelines and tools to avoid the reinvention of the wheel in life sciences. Community builds modules than can be reused by anyone.
* [BioCocoa](http://www.bioinformatics.org/biococoa/wiki/pmwiki.php) - BioCocoa is an open source OpenStep (GNUstep/Cocoa) framework for bioinformatics written in Objective-C. [Dead project].
* [iBio](https://github.com/Lizhen0909/iBio) - A Bioinformatics App for iPhone.


## Big Data
* [HDF5Kit](https://github.com/aleph7/HDF5Kit) - This is a Swift wrapper for the HDF5 file format. HDF5 is used in the scientific comunity for managing large volumes of data. The objective is to make it easy to read and write HDF5 files from Swift, including playgrounds.

# Opensource Applications
* [DeepDreamer](https://github.com/johndpope/deepdreamer) - Deep Dream application
* [DeepDreamApp](https://github.com/johndpope/DeepDreamApp) - Deep Dream Cordova app.

# Other Lists
* [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
* [Machine Learning Courses](https://github.com/prakhar1989/awesome-courses#machine-learning)
* [Awesome Data Science](https://github.com/okulbilisim/awesome-datascience)
* [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
* [Speech and language processing](https://github.com/edobashira/speech-language-processing)
* [The Rise of Chat Bots:](https://stanfy.com/blog/the-rise-of-chat-bots-useful-links-articles-libraries-and-platforms/)  Useful Links, Articles, Libraries and Platforms by Pavlo Bashmakov.

# People to follow in iOS machine learning and related topics
\(In alphabet order\).

* **Alejandro** - BrainCore and Caffe for iOS author. [Blog](http://a-coding.com). {% include icon-twitter.html username="aleph7" %}.
* **Simon Gladman** - Swift and Metal enthusiast and blogger. [Blog](http://flexmonkey.blogspot.com/). {% include icon-twitter.html username="flexmonkey" %}.
* **Andrej Karpathy** - PhD student at Stanford. ConvNetJS author. [Blog](http://karpathy.github.io/). {% include icon-twitter.html username="karpathy" %}.
* **Kalvar Lin** - wrote a set of small machine learning libraries for iOS. {% include icon-twitter.html username="ilovekalvar" %}.
* **Memkite team** - Memkite is a Deep Learning framework for iOS that can be used to support Artificial Intelligence (AI) in apps. {% include icon-twitter.html username="memkite" %}.
	* **Torbjørn Morland** {% include icon-twitter.html username="torbmorland" %}.
	* **Amund Tveit** {% include icon-twitter.html username="atveit" %}.

