# Future iOS
Curated List of 
- Machine Learning, 
- Artificial Intelligence,
- Natural Language Processing (NLP), 
- Computer Vision, 
- GPGPU, 
- Data Visualization and 
- Functional Programming 
resources for iOS. 

The list consists mostly of native libraries written in Objective-C, Swift and ports of C, C++, JavaScript libraries or libs which can be easily ported for iOS. Also links to some relevant web APIs and blog posts included.

## General-Purpose Machine Learning
https://github.com/lemire/lbimproved - DTW + kNN
https://github.com/nikolaypavlov/MLPNeuralNet - Fast multilayer perceptron neural network library for iOS and Mac OS X 
https://github.com/devongovett/SNNeuralNet

iOS & OS X framework for Shark: C++ Machine Learning Library
SHARK provides libraries for the design of adaptive systems, including methods for linear and nonlinear optimization (e.g., evolutionary and gradient-based algorithms), kernel-based algorithms and neural networks, and other machine learning techniques.
https://cocoapods.org/pods/Shark-SDK
http://sourceforge.net/projects/shark-project/

https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork
https://cocoapods.org/pods/KRBPN
https://github.com/Kalvar/ios-BPN-NeuralNetwork
https://cocoapods.org/pods/FANN

https://www.ibmdw.net/watson/
https://github.com/jetpacapp/DeepBeliefSDK
https://github.com/scottsievert/swix
https://github.com/karpathy/convnetjs
https://github.com/harthur/brain

## Game AI
http://www.raywenderlich.com/24824/introduction-to-ai-programming-for-games

## Math
https://github.com/mattt/Surge

## Natural Language Processing

https://github.com/ayanonagon/Parsimmon
NSLinguistic​Tagger

http://nshipster.com/nslinguistictagger/

http://www.alchemyapi.com/

Twitter text

The library includes methods for extracting user names, mentions headers, hashtags, and more – all the tweet specific language syntax you could ever want.

https://github.com/twitter/twitter-text-objc

An Objective-C implementation of Twitter's text processing library

https://code.google.com/p/word2vec/ - Original C implementation of Word2Vec Deep Learning algorithm. Works on iPhone like a charm.

## Speech Recognition and Generation

https://github.com/tryolabs/TLSphinx
https://github.com/vimalmurugan89/MVSpeechSynthesizer
https://tech.yandex.com/speechkit/mobilesdk/

## Computer Vision
http://maniacdev.com/2011/07/tutorial-using-and-building-opencv-open-computer-vision-on-ios-devices
https://github.com/gali8/Tesseract-OCR-iOS
Tesseract

http://lois.di-qual.net/blog/install-and-use-tesseract-on-ios-with-tesseract-ios/

https://github.com/ldiqual/tesseract-ios-lib

https://github.com/ldiqual/tesseract-ios

https://github.com/gali8/Tesseract-OCR-iOS

https://github.com/robmathews/OCR-iOS-Example

http://opencv.org/
https://github.com/foundry/OpenCVSwiftStitch

## GPGPU
https://github.com/linusyang/opencl-test-ios

http://ciechanowski.me/blog/2014/01/05/exploring_gpgpu_on_ios/

Exploring GPGPU on iOS

https://github.com/Ciechan/Exploring-GPGPU-on-iOS

https://github.com/BradLarson/GPUImage


http://www.sunsetlakesoftware.com/2010/10/22/gpu-accelerated-video-processing-mac-and-ios

https://developer.apple.com/library/ios/documentation/3ddrawing/conceptual/opengles_programmingguide/ConcurrencyandOpenGLES/ConcurrencyandOpenGLES.html

http://stackoverflow.com/questions/10704916/opencv-on-ios-gpu-usage

## Data Visualization
https://github.com/danielgindi/ios-charts
https://github.com/core-plot/core-plot
https://github.com/sxyx2008/awesome-ios-chart
https://github.com/kubatru/JTChartView

## Functional Programming
https://github.com/typelift/Swiftz
https://github.com/typelift/SwiftCheck - QuickCheck for Swift 
https://github.com/mxcl/PromiseKit
https://github.com/oisdk/SwiftSequence
https://github.com/ReactiveCocoa/ReactiveCocoa

## Databases

    Realm - The alternative to CoreData and SQLite: Simple, modern and fast.
    YapDatabase - YapDatabase is an extensible database for iOS & Mac.
    Couchbase Mobile - Couchbase document store for mobile with cloud sync.
    FMDB - A Cocoa / Objective-C wrapper around SQLite.

<!-- /MarkdownTOC -->

<a name="c" />
## C

<a name="c-general-purpose" />
#### General-Purpose Machine Learning
* [Recommender](https://github.com/GHamrouni/Recommender) - A C library for product recommendations/suggestions using collaborative filtering (CF).


<a name="c-cv" />
#### Computer Vision

* [CCV](https://github.com/liuliu/ccv) - C-based/Cached/Core Computer Vision Library, A Modern Computer Vision Library
* [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has Matlab toolbox

<a name="cpp" />
## C++

<a name="cpp-cv" />
#### Computer Vision

* [OpenCV](http://opencv.org) - OpenCV has C++, C, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS.
* [DLib](http://dlib.net/imaging.html) - DLib has C++ and Python interfaces for face detection and training general object detectors.
* [EBLearn](http://eblearn.sourceforge.net/) - Eblearn is an object-oriented C++ library that implements various machine learning models
* [VIGRA](https://github.com/ukoethe/vigra) - VIGRA is a generic cross-platform C++ computer vision and machine learning library for volumes of arbitrary dimensionality with Python bindings.

<a name="cpp-general-purpose" />
#### General-Purpose Machine Learning

* [MLPack](http://www.mlpack.org/) - A scalable C++ machine learning library
* [DLib](http://dlib.net/ml.html) - A suite of ML tools designed to be easy to imbed in other applications
* [encog-cpp](https://code.google.com/p/encog-cpp/)
* [shark](http://image.diku.dk/shark/sphinx_pages/build/html/index.html)
* [Vowpal Wabbit (VW)](https://github.com/JohnLangford/vowpal_wabbit/wiki) - A fast out-of-core learning system.
* [sofia-ml](https://code.google.com/p/sofia-ml/) - Suite of fast incremental algorithms.
* [Shogun](https://github.com/shogun-toolbox/shogun) - The Shogun Machine Learning Toolbox
* [Caffe](http://caffe.berkeleyvision.org)  - A deep learning framework developed with cleanliness, readability, and speed in mind. [DEEP LEARNING]
* [CXXNET](https://github.com/antinucleon/cxxnet) - Yet another deep learning framework with less than 1000 lines core code [DEEP LEARNING]
* [XGBoost](https://github.com/tqchen/xgboost) - A parallelized optimized general purpose gradient boosting library.
* [CUDA](https://code.google.com/p/cuda-convnet/) - This is a fast C++/CUDA implementation of convolutional [DEEP LEARNING]
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

* [Convnet.js](http://cs.stanford.edu/people/karpathy/convnetjs/) - ConvNetJS is a Javascript library for training Deep Learning models[DEEP LEARNING]
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
* [Brain](https://github.com/harthur/brain) - Neural networks in JavaScript
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

<a name="objectivec">
## Objective C

<a name="objectivec-general-purpose">
### General-Purpose Machine Learning

* [MLPNeuralNet](https://github.com/nikolaypavlov/MLPNeuralNet) - Fast multilayer perceptron neural network library for iOS and Mac OS X. MLPNeuralNet predicts new examples by trained neural network. It is built on top of the Apple's Accelerate Framework, using vectorized operations and hardware acceleration if available.
* [MAChineLearning](https://github.com/gianlucabertani/MAChineLearning) - An Objective-C multilayer perceptron library, with full support for training through backpropagation. Implemented using vDSP and vecLib, it's 20 times faster than its Java equivalent. Includes sample code for use from Swift.
* [BPN-NeuralNetwork](https://github.com/Kalvar/ios-BPN-NeuralNetwork) - It implemented 3 layers neural network ( Input Layer, Hidden Layer and Output Layer ) and it named Back Propagation Neural Network (BPN). This network can be used in products recommendation, user behavior analysis, data mining and data analysis.
* [Multi-Perceptron-NeuralNetwork](https://github.com/Kalvar/ios-Multi-Perceptron-NeuralNetwork) - it implemented multi-perceptrons neural network (ニューラルネットワーク) based on Back Propagation Neural Network (BPN) and designed unlimited-hidden-layers.
* [KRHebbian-Algorithm](https://github.com/Kalvar/ios-KRHebbian-Algorithm) - It is a non-supervisor and self-learning algorithm (adjust the weights) in neural network of Machine Learning.
* [KRKmeans-Algorithm](https://github.com/Kalvar/ios-KRKmeans-Algorithm) - It implemented K-Means the clustering and classification algorithm. It could be used in data mining and image compression.
* [KRFuzzyCMeans-Algorithm](https://github.com/Kalvar/ios-KRFuzzyCMeans-Algorithm) - It implemented Fuzzy C-Means (FCM) the fuzzy clustering / classification algorithm on Machine Learning. It could be used in data mining and image compression.



