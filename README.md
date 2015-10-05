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

## General-Purpose Machine Learning

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

### APIs

* [IBM Watson](https://www.ibmdw.net/watson/)
* [Microsoft Project Oxford](https://www.projectoxford.ai/)
* [PredictionIO](https://prediction.io/)
	* [Swift SDK](https://github.com/minhtule/PredictionIO-Swift-SDK)
	* [Tapster iOS Demo](https://github.com/minhtule/Tapster-iOS-Demo) - This demo demonstrates how to use the PredictionIO Swift SDK to integrate an iOS app with a PredictionIO engine to make your mobile app more interesting.
	* [Tutorial](https://github.com/minhtule/Tapster-iOS-Demo/blob/master/TUTORIAL.md) on using Swift with PredictionIO.
* [Google Prediction engine](https://cloud.google.com/prediction/docs)
	* [Objective-C API](https://code.google.com/p/google-api-objectivec-client/wiki/Introduction)


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
* [Wit.AI](https://wit.ai/)
* [Parsimmon](https://github.com/ayanonagon/Parsimmon)
* [NSLinguisticTagger](http://nshipster.com/nslinguistictagger/)
* [AlchemyAPI](http://www.alchemyapi.com/)

###Computational Semantics
* [Word2Vec](https://code.google.com/p/word2vec/) - Original C implementation of Word2Vec Deep Learning algorithm. Works on iPhone like a charm.

### Text Mining
* [Twitter text](https://github.com/twitter/twitter-text-objc) - 
An Objective-C implementation of Twitter's text processing library. The library includes methods for extracting user names, mentions headers, hashtags, and more – all the tweet specific language syntax you could ever want.

### Translation
* [Google Translate API](https://cloud.google.com/translate/docs)

### Speech Recognition (TTS) and Generation (STT)
* [TLSphinx](https://github.com/tryolabs/TLSphinx), [Tutorial](http://blog.tryolabs.com/2015/06/15/tlsphinx-automatic-speech-recognition-asr-in-swift/)
* [MVSpeechSynthesizer](https://github.com/vimalmurugan89/MVSpeechSynthesizer)
* [Yandex SpeechKit](https://tech.yandex.com/speechkit/mobilesdk/) for Russian language
* [OpenEars™: free speech recognition and speech synthesis for the iPhone](http://www.politepix.com/openears/) - OpenEars™ makes it simple for you to add offline speech recognition and synthesized speech/TTS to your iPhone app quickly and easily. It lets everyone get the great results of using advanced speech UI concepts like statistical language models and finite state grammars in their app, but with no more effort than creating an NSArray or NSDictionary. [Tutorial (Russian)](http://habrahabr.ru/post/237589/)

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

* [**Abbyy OCR** SDK](http://www.abbyy.com/mobile-ocr/iphone-ocr/)
* [ocrad.js](https://github.com/antimatter15/ocrad.js) - JS OCR



## GPGPU
### Articles
* [OpenCL for iOS](https://github.com/linusyang/opencl-test-ios) - just a test.
* Exploring GPGPU on iOS. 
	* [Article](http://ciechanowski.me/blog/2014/01/05/exploring_gpgpu_on_ios/) 
	* [Code](https://github.com/Ciechan/Exploring-GPGPU-on-iOS
)
* [GPUImage](https://github.com/BradLarson/GPUImage) is a GPU-accelerated image processing library.

* GPU-accelerated video processing for Mac and iOS. [Article](http://www.sunsetlakesoftware.com/2010/10/22/gpu-accelerated-video-processing-mac-and-ios0).

* https://developer.apple.com/library/ios/documentation/3ddrawing/conceptual/opengles_programmingguide/ConcurrencyandOpenGLES/ConcurrencyandOpenGLES.html

* http://stackoverflow.com/questions/10704916/opencv-on-ios-gpu-usage

#### Metal
* Simon's Gladman \(aka flexmonkey\) [blog](http://flexmonkey.blogspot.com/)
	* [talk on iOS GPU programming](https://realm.io/news/altconf-simon-gladman-ios-gpu-programming-with-swift-metal/) with Swift and Metal at Realm Altconf.
	* https://github.com/FlexMonkey/MetalReactionDiffusion
	* https://github.com/FlexMonkey/ParticleLab


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

<!-- /MarkdownTOC -->
# Unsorted links from some other lists:
// TODO: Figure out which of libraries are iOS compatible.
<a name="c" />
## C

<a name="c-cv" />
#### Computer Vision

* [CCV](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library
* [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has Matlab toolbox

<a name="cpp" />
## C++

<a name="cpp-cv" />
#### Computer Vision

* [DLib](http://dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
* [EBLearn](http://eblearn.sourceforge.net/) - Eblearn is an object-oriented C++ library that implements various machine learning models
* [VIGRA](https://github.com/ukoethe/vigra) - VIGRA is a generic cross-platform C++ computer vision and machine learning library for volumes of arbitrary dimensionality with Python bindings.

<a name="cpp-general-purpose" />
#### General-Purpose Machine Learning

* [MLPack](http://www.mlpack.org/) - A scalable C++ machine learning library
* [DLib](http://dlib.net/ml.html) - A suite of ML tools designed to be easy to imbed in other applications
* [encog-cpp](https://code.google.com/p/encog-cpp/)
* [Vowpal Wabbit (VW)](https://github.com/JohnLangford/vowpal_wabbit/wiki) - A fast out-of-core learning system.
* [sofia-ml](https://code.google.com/p/sofia-ml/) - Suite of fast incremental algorithms.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox
* [CXXNET](https://github.com/antinucleon/cxxnet) - Yet another deep learning framework with less than 1000 lines core code [DEEP LEARNING]
* [XGBoost](https://github.com/tqchen/xgboost) - A parallelized optimized general purpose gradient boosting library.
* [Stan](http://mc-stan.org/) - A probabilistic programming language implementing full Bayesian statistical inference with Hamiltonian Monte Carlo sampling
* [BanditLib](https://github.com/jkomiyama/banditlib) - A simple Multi-armed Bandit library.
* [Timbl](http://ilk.uvt.nl/timbl) - A software package/C++ library implementing several memory-based learning algorithms, among which IB1-IG, an implementation of k-nearest neighbor classification, and IGTree, a decision-tree approximation of IB1-IG. Commonly used for NLP.

<a name="cpp-nlp" />
#### Natural Language Processing
* [MIT Information Extraction Toolkit](https://github.com/mit-nlp/MITIE) - C, C++, and Python tools for named entity recognition and relation extraction
* [CRF++](https://taku910.github.io/crfpp/) - Open source implementation of Conditional Random Fields (CRFs) for segmenting/labeling sequential data & other Natural Language Processing tasks.
* [BLLIP Parser](http://github.com/BLLIP/bllip-parser) - BLLIP Natural Language Parser (also known as the Charniak-Johnson parser)
* [colibri-core](https://github.com/proycon/colibri-core) - C++ library, command line tools, and Python binding for extracting and working with basic linguistic constructions such as n-grams and skipgrams in a quick and memory-efficient way.
* [ucto](https://github.com/proycon/ucto) - Unicode-aware regular-expression based tokenizer for various languages. Tool and C++ library. Supports FoLiA format.
* [libfolia](https://github.com/proycon/libfolia) - C++ library for the [FoLiA format](https://proycon.github.io/folia)
* [frog](https://github.com/proycon/frog) - Memory-based NLP suite developed for Dutch: PoS tagger, lemmatiser, dependency parser, NER, shallow parser, morphological analyzer.
* [MeTA](https://github.com/meta-toolkit/meta) - [MeTA : ModErn Text Analysis](https://meta-toolkit.org/) is a C++ Data Sciences Toolkit that facilitates mining big text data.

#### Speech Recognition
* [Kaldi](http://kaldi.sourceforge.net/) - Kaldi is a toolkit for speech recognition written in C++ and licensed under the Apache License v2.0. Kaldi is intended for use by speech recognition researchers.

<a name="cpp-sequence" />
#### Sequence Analysis
* [ToPS](https://github.com/ayoshiaki/tops) - This is an objected-oriented framework that facilitates the integration of probabilistic models for sequences over a user defined alphabet.

<a name="javascript" />
## Javascript

<a name="javascript-nlp" />
#### Natural Language Processing

* [Twitter-text](https://github.com/twitter/twitter-text) - A JavaScript implementation of Twitter's text processing library
* [NLP.js](https://github.com/nicktesla/nlpjs) - NLP utilities in javascript and coffeescript
* [natural](https://github.com/NaturalNode/natural) - General natural language facilities for node
* [Knwl.js](https://github.com/loadfive/Knwl.js) - A Natural Language Processor in JS
* [Retext](http://github.com/wooorm/retext) - Extensible system for analyzing and manipulating natural language
* [TextProcessing](https://www.mashape.com/japerk/text-processing/support) - Sentiment analysis, stemming and lemmatization, part-of-speech tagging and chunking, phrase extraction and named entity recognition.
* [NLP Compromise](https://github.com/spencermountain/nlp_compromise) - Natural Language processing in the browser


<a name="javascript-data-analysis" />
#### Data Analysis / Data Visualization

* [D3.js](http://d3js.org/)
* [High Charts](http://www.highcharts.com/)
* [NVD3.js](http://nvd3.org/)
* [dc.js](http://dc-js.github.io/dc.js/)
* [chartjs](http://www.chartjs.org/)
* [dimple](http://dimplejs.org/)
* [amCharts](http://www.amcharts.com/)
* [D3xter](https://github.com/NathanEpstein/D3xter) - Straight forward plotting built on D3
* [statkit](https://github.com/rigtorp/statkit) - Statistics kit for JavaScript
* [science.js](https://github.com/jasondavies/science.js/) - Scientific and statistical computing in JavaScript.
* [Z3d](https://github.com/NathanEpstein/Z3d) - Easily make interactive 3d plots built on Three.js
* [Sigma.js](http://sigmajs.org/) - JavaScript library dedicated to graph drawing.
* [C3.js](http://c3js.org/)- customizable library based on D3.js for easy chart drawing. 

<a name="javascript-general-purpose" />
#### General-Purpose Machine Learning

* [Clusterfck](http://harthur.github.io/clusterfck/) - Agglomerative hierarchical clustering implemented in Javascript for Node.js and the browser
* [Clustering.js](https://github.com/tixz/clustering.js) - Clustering algorithms implemented in Javascript for Node.js and the browser
* [Decision Trees](https://github.com/serendipious/nodejs-decision-tree-id3) - NodeJS Implementation of Decision Tree using ID3 Algorithm
* [figue](http://code.google.com/p/figue/) - K-means, fuzzy c-means and agglomerative clustering
* [Node-fann](https://github.com/rlidwka/node-fann) - FANN (Fast Artificial Neural Network Library) bindings for Node.js
* [Kmeans.js](https://github.com/tixz/kmeans.js) - Simple Javascript implementation of the k-means algorithm, for node.js and the browser
* [LDA.js](https://github.com/primaryobjects/lda) - LDA topic modeling for node.js
* [Learning.js](https://github.com/yandongliu/learningjs) - Javascript implementation of logistic regression/c4.5 decision tree
* [Machine Learning](http://joonku.com/project/machine_learning) - Machine learning library for Node.js
* [mil-tokyo](https://github.com/mil-tokyo) - List of several machine learning libraries
* [Node-SVM](https://github.com/nicolaspanel/node-svm) - Support Vector Machine for nodejs
* [Bayesian-Bandit](https://github.com/omphalos/bayesian-bandit.js) - Bayesian bandit implementation for Node and the browser.
* [Synaptic](https://github.com/cazala/synaptic) - Architecture-free neural network library for node.js and the browser
* [kNear](https://github.com/NathanEpstein/kNear) - JavaScript implementation of the k nearest neighbors algorithm for supervised learning
* [NeuralN](https://github.com/totemstech/neuraln) - C++ Neural Network library for Node.js. It has advantage on large dataset and multi-threaded training.
* [kalman](https://github.com/itamarwe/kalman) - Kalman filter for Javascript.
* [shaman](https://github.com/dambalah/shaman) - node.js library with support for both simple and multiple linear regression.

<a name="javascript-misc" />
#### Misc

* [sylvester](https://github.com/jcoglan/sylvester) - Vector and Matrix math for JavaScript.
* [simple-statistics](https://github.com/tmcw/simple-statistics) - A JavaScript implementation of descriptive, regression, and inference statistics. Implemented in literate JavaScript with no dependencies, designed to work in all modern browsers (including IE) as well as in node.js.
* [regression-js](https://github.com/Tom-Alexander/regression-js) - A javascript library containing a collection of least squares fitting methods for finding a trend in a set of data.
* [Lyric](https://github.com/flurry/Lyric) - Linear Regression library.
* [GreatCircle](https://github.com/mwgg/GreatCircle) - Library for calculating great circle distance.

## Data Visualization
*Data visualization tools for the web.*

* [d3](https://github.com/mbostock/d3) - A JavaScript visualization library for HTML and SVG.
  * [metrics-graphics](https://github.com/mozilla/metrics-graphics) - A library optimized for concise, principled data graphics and layouts.
* [pykcharts.js](https://github.com/pykih/PykCharts.js) - Well designed d3.js charting without the complexity of d3.js.
* [three.js](https://github.com/mrdoob/three.js) - JavaScript 3D library.
* [Chart.js](https://github.com/nnnick/Chart.js) - Simple HTML5 Charts using the <canvas> tag.
* [paper.js](https://github.com/paperjs/paper.js) - The Swiss Army Knife of Vector Graphics Scripting – Scriptographer ported to JavaScript and the browser, using HTML5 Canvas.
* [fabric.js](https://github.com/kangax/fabric.js) - Javascript Canvas Library, SVG-to-Canvas (& canvas-to-SVG) Parser.
* [peity](https://github.com/benpickles/peity) - Progressive <svg> bar, line and pie charts.
* [raphael](https://github.com/DmitryBaranovskiy/raphael) - JavaScript Vector Library.
* [echarts](https://github.com/ecomfe/echarts) - Enterprise Charts.
* [vis](https://github.com/almende/vis) - Dynamic, browser-based visualization library.
* [two.js](https://github.com/jonobr1/two.js) - A renderer agnostic two-dimensional drawing api for the web.
* [g.raphael](https://github.com/DmitryBaranovskiy/g.raphael) - Charts for Raphaël.
* [sigma.js](https://github.com/jacomyal/sigma.js) - A JavaScript library dedicated to graph drawing.
* [arbor](https://github.com/samizdatco/arbor) - A graph visualization library using web workers and jQuery.
* [cubism](https://github.com/square/cubism) - A D3 plugin for visualizing time series.
* [dc.js](https://github.com/dc-js/dc.js) - Multi-Dimensional charting built to work natively with crossfilter rendered with d3.js
* [vega](https://github.com/trifacta/vega) - A visualization grammar.
* [envisionjs](https://github.com/HumbleSoftware/envisionjs) - Dynamic HTML5 visualization.
* [rickshaw](https://github.com/shutterstock/rickshaw) - JavaScript toolkit for creating interactive real-time graphs.
* [flot](https://github.com/flot/flot) - Attractive JavaScript charts for jQuery.
* [morris.js](https://github.com/morrisjs/morris.js) - Pretty time-series line graphs.
* [nvd3](https://github.com/novus/nvd3) - Build re-usable charts and chart components for d3.js
* [svg.js](https://github.com/wout/svg.js) - A lightweight library for manipulating and animating SVG.
* [heatmap.js](https://github.com/pa7/heatmap.js) - JavaScript Library for HTML5 canvas based heatmaps.
* [jquery.sparkline](https://github.com/gwatts/jquery.sparkline) - A plugin for the jQuery javascript library to generate small sparkline charts directly in the browser.
* [xCharts](https://github.com/tenxer/xCharts) - A D3-based library for building custom charts and graphs.
* [trianglify](https://github.com/qrohlf/trianglify) - Low poly style background generator with d3.js
* [d3-cloud](https://github.com/jasondavies/d3-cloud) - Create word clouds in JavaScript.
* [d4](https://github.com/heavysixer/d4) - A friendly reusable charts DSL for D3.
* [dimple.js](http://dimplejs.org) -  Easy charts for business analytics powered by d3
* [chartist-js](https://github.com/gionkunz/chartist-js) - Simple responsive charts.
* [epoch](https://github.com/fastly/epoch) - A general purpose real-time charting library.
* [c3](https://github.com/masayuki0812/c3) - D3-based reusable chart library.
* [BabylonJS](https://github.com/BabylonJS/Babylon.js) - A framework for building 3D games with HTML 5 and WebGL.

There're also some great commercial libraries, like [amchart](http://www.amcharts.com/), [plotly](https://www.plot.ly/), and [highchart](http://www.highcharts.com/).

## Numerical - C
<li><a href="https://github.com/b-k/apophenia" rel="nofollow">apophenia</a> - A library for statistical and scientific computing. <a href="http://www.gnu.org/licenses/old-licenses/gpl-2.0.html" rel="nofollow">GNU GPL2.1</a> with some <a href="https://github.com/b-k/apophenia/blob/master/install/COPYING2" rel="nofollow">exceptions</a>.<br></li>
<li><a href="http://math-atlas.sourceforge.net/" rel="nofollow">ATLAS</a> - Automatically Tuned Linear Algebra Software. <a href="http://directory.fsf.org/wiki/License:BSD_3Clause" rel="nofollow">3-clause BSD</a>.<br></li>
<li><a href="http://www.netlib.org/blas/" rel="nofollow">BLAS</a> - Basic Linear Algebra Subprograms; a set of routines that provide vector and matrix operations. <a href="http://www.netlib.org/blas/#_licensing" rel="nofollow">BLAS license</a><br></li>
<li><a href="http://lipforge.ens-lyon.fr/www/crlibm/index.html" rel="nofollow">CRlibm</a> - Correctly Rounded mathematical library; a modern implementation of a range of numeric routines. <a href="http://www.gnu.org/licenses/lgpl.html" rel="nofollow">GN LGPL3</a>.<br></li>
<li><a href="http://www.feynarts.de/cuba/" rel="nofollow">Cuba</a> - A library for multidimensional numerical integration. <a href="http://www.gnu.org/licenses/lgpl.html" rel="nofollow">GNU LGPL3</a>.<br></li>
<li><a href="http://www.fftw.org/" rel="nofollow">FFTW</a> - The Fastest Fourier Transform in the West; a highly-optimized fast Fourier transform routine. <a href="http://www.gnu.org/licenses/old-licenses/gpl-2.0.html" rel="nofollow">GNU GPL2.1</a>.<br></li>
<li><a href="http://flintlib.org/" rel="nofollow">FLINT</a> - Fast Library for Number Theory; a library supporting arithmetic with numbers, polynomials, power series and matrices, among others. <a href="http://www.gnu.org/licenses/old-licenses/gpl-2.0.html" rel="nofollow">GNU GPL2.1</a>.<br></li>
<li><a href="https://gnu.org/software/glpk/" rel="nofollow">GLPK</a> - GNU Linear Programming Kit; a package designed for solving large-scale linear programming, mixed integer programming and other related problems. <a href="http://www.gnu.org/licenses/gpl.html" rel="nofollow">GNU GPL3</a>.<br></li>
<li><a href="https://gmplib.org/" rel="nofollow">GMP</a> - GNU Multple Precision Arithmetic Library; a library for arbitrary-precision arithmetic. <a href="http://www.gnu.org/licenses/old-licenses/gpl-2.0.html" rel="nofollow">GNU GPL2.1</a> and <a href="http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html" rel="nofollow">GNU LGPL2.1</a>.<br></li>
<li><a href="http://www.multiprecision.org/index.php?prog=mpc&amp;page=home" rel="nofollow">GNU MPC</a> - A library for complex number arithmetic. <a href="http://www.gnu.org/licenses/lgpl.html" rel="nofollow">GNU LGPL3</a>.<br></li>
<li><a href="http://mpfr.loria.fr/index.html" rel="nofollow">GNU MPFR</a> - A library for arbitrary-precision floating-point arithmetic. <a href="http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html" rel="nofollow">GNU LGPL2.1</a>.<br></li>
<li><a href="https://gnu.org/software/mpria/" rel="nofollow">GNU MPRIA</a> - A portable mathematics library for multi-precision rational interval arithmetic. <a href="http://www.gnu.org/licenses/gpl.html" rel="nofollow">GNU GPL3</a>.<br></li>
<li><a href="http://www.gnu.org/software/gsl/" rel="nofollow">GSL</a> - The GNU Scientific Library; a sophisticated numerical library. <a href="http://www.gnu.org/licenses/gpl.html" rel="nofollow">GNU GPL3</a>.<br></li>
<li><a href="http://sourceforge.net/projects/kissfft/" rel="nofollow">KISS FFT</a> - A very simple fast Fourier transform library. <a href="http://directory.fsf.org/wiki/License:BSD_3Clause" rel="nofollow">3-clause BSD</a>.<br></li>
<li><a href="http://www.netlib.org/lapack/lapacke.html" rel="nofollow">LAPACKE</a> - A C interface to <a href="http://www.netlib.org/lapack/" rel="nofollow">LAPACK</a>. <a href="http://directory.fsf.org/wiki/License:BSD_3Clause" rel="nofollow">3-clause BSD</a>.<br></li>
<li><a href="http://pari.math.u-bordeaux.fr/" rel="nofollow">PARI/GP</a> - A computer algebra system for number theory; includes a compiler to C. <a href="http://www.gnu.org/licenses/gpl.html" rel="nofollow">GNU GPL3</a>.<br></li>
<li><a href="http://www.mcs.anl.gov/petsc/" rel="nofollow">PETSc</a> - A suite of data structures and routines for scalable parallel solution of scientific applications modelled by partial differential equations. <a href="http://directory.fsf.org/wiki?title=License:FreeBSD" rel="nofollow">FreeBSD</a>.<br></li>
<li><a href="http://slepc.upv.es/" rel="nofollow">SLEPc</a> - A software library for the solution of large, sparse eigenvalue problems on parallel computers. <a href="http://www.gnu.org/licenses/lgpl.html" rel="nofollow">GNU LGPL3</a>.<br></li>
<li><a href="http://www.yeppp.info/" rel="nofollow">Yeppp!</a> - Very fast, SIMD-optimized mathematical library. <a href="http://directory.fsf.org/wiki/License:BSD_3Clause" rel="nofollow">3-clause BSD</a>.<br></li>

# C++
## Artificial Intelligence

* [btsk](https://github.com/aigamedev/btsk) - Game Behavior Tree Starter Kit. [zlib]
* [Evolving Objects](http://eodev.sourceforge.net/) - A template-based, ANSI-C++ evolutionary computation library which helps you to write your own stochastic optimization algorithms insanely fast. [LGPL]
* [Neu](https://github.com/andrometa/neu) - A C++ 11 framework, collection of programming languages, and multipurpose software system designed for: the creation of artificial intelligence applications. [BSD]

## Biology
*Bioinformatics, Genomics, Biotech*

* [libsequence](http://molpopgen.github.io/libsequence/) - A C++ library for representing and analyzing population genetics data. [GPL]
* [SeqAn](http://www.seqan.de/) - Algorithms and data structures for the analysis of sequences with the focus on biological data. [BSD/3-clause]
* [Vcflib](https://github.com/ekg/vcflib) - A C++ library for parsing and manipulating VCF files. [MIT]
* [Wham](https://github.com/jewmanchue/wham) - Structural variants (SVs) in Genomes by directly applying association tests to BAM files. [MIT]

## Machine Learning

* [CCV](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library. [BSD]
* [MeTA](https://github.com/meta-toolkit/meta) - A modern C++ data sciences toolkit. [MIT] [website](https://meta-toolkit.org/)
* [Minerva](https://github.com/minerva-developers/minerva) - A fast and flexible system for deep learning. [Apache2]
* [mlpack](http://www.mlpack.org/) - A scalable c++ machine learning library. [LGPLv3]
* [Recommender](https://github.com/GHamrouni/Recommender) - C library for product recommendations/suggestions using collaborative filtering (CF). [BSD]
* [SHOGUN](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox. [GPLv3]
* [sofia-ml](https://code.google.com/p/sofia-ml/) - The suite of fast incremental algorithms for machine learning. [Apache2]

## Math

* [Apophenia](https://github.com/b-k/apophenia) - A C library for statistical and scientific computing [GPL2]
* [Armadillo](http://arma.sourceforge.net/) - A high quality C++ linear algebra library, aiming towards a good balance between speed and ease of use. The syntax (API) is deliberately similar to Matlab. [MPL2]
* [blaze](https://code.google.com/p/blaze-lib/) - high-performance C++ math library for dense and sparse arithmetic. [BSD]
* [Boost.Multiprecision](http://www.boost.org/doc/libs/master/libs/multiprecision/doc/html/index.html) - provides higher-range/precision integer, rational and floating-point types in C++, header-only or with GMP/MPFR/LibTomMath backends. [Boost]
* [ceres-solver](http://ceres-solver.org/) - C++ library for modeling and solving large complicated nonlinear least squares problems from google. [BSD]
* [CGal](http://www.cgal.org/) - Collection of efficient and reliable geometric algorithms. [LGPL&GPL]
* [cml](http://cmldev.net/) - free C++ math library for games and graphics. [Boost]
* [Eigen](http://eigen.tuxfamily.org/) - A high-level C++ library of template headers for linear algebra, matrix and vector operations, numerical solvers and related algorithms. [MPL2]
* [GLM](https://github.com/g-truc/glm) - Header-only C++ math library that matches and inter-operates with OpenGL's GLSL math. [MIT]
* [GMTL](http://ggt.sourceforge.net/) - Graphics Math Template Library is a collection of tools implementing Graphics primitives in generalized ways. [GPL2]
* [GMP](https://gmplib.org/) - A C/C++ library for arbitrary precision arithmetic, operating on signed integers, rational numbers, and floating-point numbers. [LGPL3 & GPL2]
* [MIRACL](https://github.com/CertiVox/MIRACL) - A Multiprecision Integer and Rational Arithmetic Cryptographic Library. [AGPL]
* [LibTomMath](https://github.com/libtom/libtommath) - A free open source portable number theoretic multiple-precision integer library written entirely in C. [PublicDomain & WTFPL] [website](http://www.libtom.net/)
* [QuantLib](https://github.com/lballabio/quantlib) - A free/open-source library for quantitative finance. [Modified BSD] [website](http://quantlib.org/)

## Scientific Computing

* [FFTW](http://www.fftw.org/) - A C library for computing the DFT in one or more dimensions. [GPL]
* [GSL](http://www.gnu.org/software/gsl/) - GNU scientific library. [GPL]
