# Future iOS - tools and resources to create really smart iOS applications.

[![Join the chat at https://gitter.im/alexsosn/iOS_ML](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/alexsosn/iOS_ML?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Curated List of resources for iOS developers in following topics: 

- Machine Learning, 
- Artificial Intelligence,
- Natural Language Processing (NLP), 
- Computer Vision, 
- General-Purpose GPU Computing (GPGPU), 
- Data Visualization,
- Bioinformatics

Most of the de-facto standard tools from domains listed above are written in iOS-unfiendly languages (Python/Java/R/Matlab) so find something appropriate for your iOS application may be a difficult task.
This list consists mainly of native libraries written in Objective-C, Swift and ports of C, C++, JavaScript libraries or libs which can be easily ported to iOS. Also links to some relevant web APIs, blog posts, videos and learning materials included.

## Where to learn about machine learning and other related staff in general
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
* [AlchemyAPI](http://www.alchemyapi.com/) - Semantic Text Analysis APIs Using Natural Language Processing. Now part of IBM Watson.
* [**Microsoft** Project Oxford](https://www.projectoxford.ai/)
* [**Google** Prediction engine](https://cloud.google.com/prediction/docs)
	* [Objective-C API](https://code.google.com/p/google-api-objectivec-client/wiki/Introduction)
* [Google Translate API](https://cloud.google.com/translate/docs)
* [**Amazon** Machine Learning](http://aws.amazon.com/documentation/machine-learning/) - Amazon ML is a cloud-based service for developers. It provides visualization tools to create machine learning models. Obtain predictions for application using APIs. [iOS developer guide](https://docs.aws.amazon.com/mobile/sdkforios/developerguide/getting-started-machine-learning.html).
* [**PredictionIO**](https://prediction.io/) - opensource machine learning server for developers and ML engineers. Built on Apache Spark, HBase and Spray.
	* [Swift SDK](https://github.com/minhtule/PredictionIO-Swift-SDK)
	* [Tapster iOS Demo](https://github.com/minhtule/Tapster-iOS-Demo) - This demo demonstrates how to use the PredictionIO Swift SDK to integrate an iOS app with a PredictionIO engine to make your mobile app more interesting.
	* [Tutorial](https://github.com/minhtule/Tapster-iOS-Demo/blob/master/TUTORIAL.md) on using Swift with PredictionIO.
* [**Wit.AI**](https://wit.ai/) - NLP API
* [**Yandex** SpeechKit](https://tech.yandex.com/speechkit/mobilesdk/) Text-to-speech and speech-to-text for Russian language. iOS SDK available.
* [**Abbyy** OCR SDK](http://www.abbyy.com/mobile-ocr/iphone-ocr/)


## General-Purpose Machine Learning Libraries

* [MLPNeuralNet](https://github.com/nikolaypavlov/MLPNeuralNet) - Fast multilayer perceptron neural network library for iOS and Mac OS X. MLPNeuralNet predicts new examples by trained neural network. It is built on top of the Apple's Accelerate Framework, using vectorized operations and hardware acceleration if available.
* [MAChineLearning](https://github.com/gianlucabertani/MAChineLearning) - An Objective-C multilayer perceptron library, with full support for training through backpropagation. Implemented using vDSP and vecLib, it's 20 times faster than its Java equivalent. Includes sample code for use from Swift.
* [FANN](https://cocoapods.org/pods/FANN) - Fast Artifical Neural Network library; an implementation of neural networks.
* [lbimproved](https://github.com/lemire/lbimproved) - DTW + kNN in C
* [Recommender](https://github.com/GHamrouni/Recommender) - A C library for product recommendations/suggestions using collaborative filtering (CF).
* [SNNeuralNet](https://github.com/devongovett/SNNeuralNet) - A neural network library for Objective-C based on brain.js, for iOS and Mac OS X.
* **Shark** - provides libraries for the design of adaptive systems, including methods for linear and nonlinear optimization (e.g., evolutionary and gradient-based algorithms), kernel-based algorithms and neural networks, and other machine learning techniques. [CocoaPods](https://cocoapods.org/pods/Shark-SDK). [Official site](http://image.diku.dk/shark/sphinx_pages/build/html/index.html)
* [BPN-NeuralNetwork](https://github.com/Kalvar/ios-BPN-NeuralNetwork) - It implemented 3 layers neural network ( Input Layer, Hidden Layer and Output Layer ) and it named Back Propagation Neural Network (BPN). This network can be used in products recommendation, user behavior analysis, data mining and data analysis.
* [Multi-Perceptron-NeuralNetwork](https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork) - it implemented multi-perceptrons neural network based on Back Propagation Neural Network (BPN) and designed unlimited-hidden-layers.
* [KRHebbian-Algorithm](https://github.com/Kalvar/ios-KRHebbian-Algorithm) - It is a non-supervisor and self-learning algorithm (adjust the weights) in neural network of Machine Learning.
* [KRKmeans-Algorithm](https://github.com/Kalvar/ios-KRKmeans-Algorithm) - It implemented K-Means the clustering and classification algorithm. It could be used in data mining and image compression.
* [KRFuzzyCMeans-Algorithm](https://github.com/Kalvar/ios-KRFuzzyCMeans-Algorithm) - It implemented Fuzzy C-Means (FCM) the fuzzy clustering / classification algorithm on Machine Learning. It could be used in data mining and image compression.
* [Torch-iOS](https://github.com/clementfarabet/torch-ios) - [Torch](http://torch.ch/) port for iOS. Torch is a scientific computing framework with wide support for machine learning algorithms. One of the most popular deep learning frameworks.
* [Caffe](http://caffe.berkeleyvision.org) - A deep learning framework developed with cleanliness, readability, and speed in mind.
[GitHub](https://github.com/BVLC/caffe). [BSD]
	* C++ examples: [Classifying ImageNet](http://caffe.berkeleyvision.org/gathered/examples/cpp_classification.html), [Extracting Features](http://caffe.berkeleyvision.org/gathered/examples/feature_extraction.html)
	* [Caffe iOS sample](https://github.com/noradaiko/caffe-ios-sample)
* [Deep Belief SDK](https://github.com/jetpacapp/DeepBeliefSDK)
* [Swix](https://github.com/scottsievert/swix) - Swift implementation of NumPy.
* [Convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/) - ConvNetJS is a Javascript library for training Deep Learning models by [Andrej Karpathy](https://twitter.com/karpathy). [GitHub](https://github.com/karpathy/convnetjs)
* [Brain](https://github.com/harthur/brain) - Neural networks in JavaScript
* [April-ANN](https://github.com/pakozm/april-ann) -  APRIL-ANN toolkit (A Pattern Recognizer In Lua with ANNs). This toolkit incorporates ANN algorithms (as dropout, stacked denoising auto-encoders, convolutional NNs), with other pattern recognition methods as HMMs among others. Additionally, in experimental stage, it is possible to perform automatic differentiation, for advanced ML research. Potentially can be ported for iOS. 
	* [Official Site](http://pakozm.github.com/april-ann/). 

## AI
* [Mendel](https://github.com/saniul/Mendel) - Genetic algorithms in Swift.

### Game AI
* [Introduction to AI Programming for Games](http://www.raywenderlich.com/24824/introduction-to-ai-programming-for-games)
* [dlib](http://dlib.net/) is a library which has many useful tools including machine learning.
* [MicroPather](http://www.grinninglizard.com/MicroPather/) is a path finder and A* solver (astar or a-star) written in platform independent C++ that can be easily integrated into existing code.
* Here is a [list](http://www.ogre3d.org/tikiwiki/List+Of+Libraries#Artificial_intelligence) of some AI libraries suggested on OGRE3D website. Seems they are mostly written in C++ and at least most of them are not depended on OGRE engine itself.
* [GameplayKit Programming Guide](https://developer.apple.com/library/prerelease/ios/documentation/General/Conceptual/GameplayKit_Guide/Minmax.html#//apple_ref/doc/uid/TP40015172-CH2-SW1)

## Math
* [Surge](https://github.com/mattt/Surge) from Mattt

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

## Computer Vision
* [**Vuforia** Framework](https://developer.vuforia.com/downloads/sdk)
	* [Object detection with Vuforia tutorial.](http://habrahabr.ru/company/dataart/blog/256385/) \(Russian\)

* **[OpenCV](http://opencv.org)** - Open Source Computer Vision Library. [BSD]
	* [OpenCV crash course](http://www.pyimagesearch.com/free-opencv-crash-course/
) 
	* https://github.com/foundry/OpenCVSwiftStitch
	* http://maniacdev.com/2011/07/tutorial-using-and-building-opencv-open-computer-vision-on-ios-devices
	* ML:
	 * Normal Bayes Classifier
	 * K-Nearest Neighbour Classifier  
	 * Support Vector Machines  
	 * Expectation - Maximisation     
	 * Decision Tree 
	 * Random Trees Classifier 
	 * Extremely randomised trees Classifier  
	 * Boosted tree classifier
	 * Gradient Boosted Trees     
	 * Multi-Layer Perceptrons
	* CV:
	 * Haar-like Object Detection  
	 * Latent SVM Object Detection 
	 * HOG (Histogram-of-Oriented-Gradients) Descriptor and Object Detector
	 * [Data Matrix detection](https://en.wikipedia.org/wiki/Data_Matrix) 
	 * LINE template matching algorithm

* [trackingjs](http://trackingjs.com/) - Object tracking in JS

### Text Recognition (OCR)
* **Tesseract**
	* http://lois.di-qual.net/blog/install-and-use-tesseract-on-ios-with-tesseract-ios/
	* https://github.com/ldiqual/tesseract-ios-lib
	* https://github.com/ldiqual/tesseract-ios
	* https://github.com/gali8/Tesseract-OCR-iOS
	* https://github.com/robmathews/OCR-iOS-Example
* [ocrad.js](https://github.com/antimatter15/ocrad.js) - JS OCR


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


# Other Lists
* [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
* [Machine Learning Courses](https://github.com/prakhar1989/awesome-courses#machine-learning)
* [Awesome Data Science](https://github.com/okulbilisim/awesome-datascience)
* [Awesome Computer Vision](https://github.com/jbhuang0604/awesome-computer-vision)
* [Speech and language processing](https://github.com/edobashira/speech-language-processing)
