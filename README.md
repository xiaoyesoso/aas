Advanced Analytics with Spark Source Code
=========================================

Code to accompany [Advanced Analytics with Spark](http://shop.oreilly.com/product/0636920035091.do), 
by [Sandy Ryza](https://github.com/sryza), [Uri Laserson](https://github.com/laserson), 
[Sean Owen](https://github.com/srowen), and [Josh Wills](https://github.com/jwills).

[![Advanced Analytics with Spark](http://akamaicovers.oreilly.com/images/0636920035091/lrg.jpg)](http://shop.oreilly.com/product/0636920035091.do)

### Build

[Apache Maven](http://maven.apache.org/) 3.0.5+ and Java 7+ are required to build. From the root level of the project, run `mvn package` to compile artifacts into `target/` subdirectories beneath each chapter's directory.

### Data Sets

- Chapter 2: https://archive.ics.uci.edu/ml/machine-learning-databases/00210/
- Chapter 3: http://www-etud.iro.umontreal.ca/~bergstrj/audioscrobbler_data.html
- Chapter 4: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
- Chapter 5: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html (do _not_ use http://www.sigkdd.org/kdd-cup-1999-computer-network-intrusion-detection as the copy has a corrupted line)
- Chapter 6: http://dumps.wikimedia.org/enwiki/20150112/enwiki-20150112-pages-articles-multistream.xml.bz2
- Chapter 7: ftp://ftp.nlm.nih.gov/nlmdata/sample/medline/ (`*.gz`)
- Chapter 8: http://www.andresmh.com/nyctaxitrips/
- Chapter 9: (see `ch09-risk/data/download-all-symbols.sh` script)
- Chapter 10: ftp://ftp.ncbi.nih.gov/1000genomes/ftp/phase3/data/HG00103/alignment/HG00103.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam
- Chapter 11: https://github.com/thunder-project/thunder/tree/v0.4.1/python/thunder/utils/data/fish/tif-stack

[![Build Status](https://travis-ci.org/sryza/aas.png?branch=master)](https://travis-ci.org/sryza/aas)

#ch04 决策树，随机森林——预测森林植被类型
##数据集处理
```scala
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression._
val rawData = sc.textFile("covtype.data")
val data = rawData.map{	
	line =>
	val values = line.split(",").map( _.toDouble)
	//init返回除最后一个值外的所有值
	val featureVector = Vectors.dense(values.init)
	//决策树要求label从0开始
	val label = values.last -1
	LabeledPoint( label,featureVector)
}

val Array(trainData,cvData,testData) = data.randomSplit( Array(0.8,0.1,0.1))
trainData.cache() 
cvData.cache() //交叉检验集
testData.cache()
```

##模型训练
```scala
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

def getMetrics(model: DecisionTreeModel,dta: RDD[ LabeledPoint ]):
	MulticlassMetrics = {
		val predictionsAndLabels = data.map( example =>
		( 
			model.predict( example.features), example.label)
		)
		new MulticlassMetrics( predictionsAndLabels)
	}


val model = DecisionTree.trainClassifier(trainData,7,Map[Int,Int](),"gini",4,100)
```
决策树有训练分类模型的函数trainClassifier和回归模型的函数trainRegressor，这里我们使用trainClassifier。
我们来看看trainClassifier都需要什么参数：
```shell
scala> DecisionTree.trainClassifier
<console>:42: error: ambiguous reference to overloaded definition,
both method trainClassifier in object DecisionTree of type (input: org.apache.spark.api.java.JavaRDD[org.apache.spark.mllib.regression.LabeledPoint], numClasses: Int, categoricalFeaturesInfo: java.util.Map[Integer,Integer], impurity: String, maxDepth: Int, maxBins: Int)org.apache.spark.mllib.tree.model.DecisionTreeModel
and  method trainClassifier in object DecisionTree of type (input: org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint], numClasses: Int, categoricalFeaturesInfo: Map[Int,Int], impurity: String, maxDepth: Int, maxBins: Int)org.apache.spark.mllib.tree.model.DecisionTreeModel
match expected type ?
       DecisionTree.trainClassifier
                    ^
```
- input:数据的LabeledPoint
- numClasses：类别数量
- categoricalFeaturesInfo：
看下[Saprk官网](https://spark.apache.org/docs/latest/mllib-decision-tree.html#usage-tips)的介绍:

>categoricalFeaturesInfo: Specifies which features are categorical and how many categorical values each of those features can take. This is given as a map from feature indices to feature arity (number of categories). Any features not in this map are treated as continuous.
  
>E.g., Map(0 -> 2, 4 -> 10) specifies that feature 0 is binary (taking values 0 or 1) and that feature 4 has 10 categories (values {0, 1, ..., 9}). Note that feature indices are 0-based: features 0 and 4 are the 1st and 5th elements of an instance’s feature vector.

>Note that you do not have to specify categoricalFeaturesInfo. The algorithm will still run and may get reasonable results. However, performance should be better if categorical features are properly designated.


- impurity：不纯度的类型，有基尼不纯度——“gini”，熵——“entropy”
- maxDepth：对层数进行限制，避免过拟合
- maxBins：决策规则集，可以理解成是决策树的孩子节点的数量

##性能评估
```scala
import org.apache.spark.mllib.evaluation._
val metrics = getMetrics(model,cvData)
metrics.confusionMatrix
/*
res6: org.apache.spark.mllib.linalg.Matrix =                                    
156710.0  51350.0   203.0    0.0  0.0    0.0  3577.0
68735.0   207253.0  6883.0   0.0  42.0   0.0  388.0
0.0       5872.0    29882.0  0.0  0.0    0.0  0.0
0.0       0.0       2747.0   0.0  0.0    0.0  0.0
105.0     8702.0    557.0    0.0  129.0  0.0  0.0
0.0       4475.0    12892.0  0.0  0.0    0.0  0.0
11290.0   239.0     55.0     0.0  0.0    0.0  8926.0 
*/
```
因为一共有7种类别，所以生成的是7*7的矩阵，aij 表示实际类别是i，而被预测类别是j的次数。
```scala
metrics.precision
//res7: Double = 0.6934452300468837 
```
##决策树调优
```scala
val evaluations =
      for (impurity <- Array("gini", "entropy");
           depth    <- Array(1, 20);
           bins     <- Array(10, 300))
        yield {
          val model = DecisionTree.trainClassifier(
            trainData, 7, Map[Int,Int](), impurity, depth, bins)
          val accuracy = getMetrics(model, cvData).precision
          ((impurity, depth, bins), accuracy)
        }
evaluations.sortBy(_._2).reverse.foreach( println)
/*
((entropy,20,300),0.9380098861985638)
((gini,20,300),0.9319721451536285)
((entropy,20,10),0.9273681094366382)
((gini,20,10),0.9195954644654499)
((gini,1,10),0.633916339077334)
((gini,1,300),0.6335772755123819)
((entropy,1,300),0.48759922342395684)
((entropy,1,10),0.48759922342395684)
*/
```
- scala语法：
```scala
  for (impurity <- Array("gini", "entropy");
           depth    <- Array(1, 20);
           bins     <- Array(10, 300))
        yield {}
```
相当于关于impurity，depth，bins的三层循环。

##关于categoricalFeaturesInfo
关于categoricalFeaturesInfo这个参数，我们前面直接不设定取值个数:
```scala
Map[Int,Int]()
```
但是，我们可以参阅下[covtype.info](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info)关于数据集的描述：
>……

>Hillshade_9am                           quantitative    0 to 255 index               Hillshade index at 9am, summer solstice

>Hillshade_Noon                          quantitative    0 to 255 index               Hillshade index at noon, summer soltice

>Hillshade_3pm                           quantitative    0 to 255 index               Hillshade index at 3pm, summer solstice

>Wilderness_Area (4 binary columns)      qualitative     0 (absence) or 1 (presence)  Wilderness area designation

>Soil_Type (40 binary columns)           qualitative     0 (absence) or 1 (presence)  Soil Type designation

>……

>Wilderness Areas:  	1 -- Rawah Wilderness Area
                        2 -- Neota Wilderness Area
                        3 -- Comanche Peak Wilderness Area
                        4 -- Cache la Poudre Wilderness Area

>Soil Types:             1 to 40 : based on the USFS Ecological
                        Landtype Units (ELUs) for this study area

可知：
- 三个Hillshade都有256种取值
- Wilderness Areas 有4中类别，Soil Types 有40种。数据集中是以二元特征的形式，有4列，如取值为3，那么第三列为1，其它列都为0

###重新处理数据集
```scala
  def unencodeOneHot(rawData: RDD[String]): RDD[LabeledPoint] = {
    rawData.map { line =>
      val values = line.split(',').map(_.toDouble)
      /*我们可以从covtype.info中得知：wilderness是从第10行开始的，
        slice(10, 14) 截取 10 到 13 行
        indexOf(1.0)  返回值为1的位置编号
     */
      val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
      val soil = values.slice(14, 54).indexOf(1.0).toDouble
      val featureVector = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
      val label = values.last - 1
      LabeledPoint(label, featureVector)
    }
  }
  
    val data = unencodeOneHot(rawData)

    val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
    trainData.cache()
    cvData.cache()
    testData.cache()
```
###重新评估性能
这里进行参数设置时发现这样的错误：
```shell
java.lang.IllegalArgumentException: requirement failed: DecisionTree requires maxBins (= 40) to be at least as large as the number of values in each categorical feature, but categorical feature 6 has 256 values. Considering remove this and other categorical features with a large number of values, or add more training examples.
```
所以：***bins数量必须大于等于Max（各个feature的values数量）***
```scala
val evaluations =
      for (impurity <- Array("gini", "entropy");
      depth    <- Array(10, 20,30);
      bins     <- Array(256, 300))
      yield{
      val model = DecisionTree.trainClassifier(
      trainData,7,Map(6 -> 256,7 -> 256,8 -> 256,10 -> 4,11 -> 40),
      impurity, depth, bins)
      val accurary = getMetrics(model, cvData).precision
      (( impurity,depth,bins), accurary)
      }

evaluations.sortBy(_._2).reverse.foreach( println)
/*
((gini,30,300),0.6327390828416625)
((gini,20,300),0.6319645721602997)
((gini,10,256),0.6190078690285227)
((gini,30,256),0.6165724632193483)
((gini,20,256),0.6149373851142489)
((gini,10,300),0.596522963381135)
((entropy,30,256),0.5868863293701335)
((entropy,20,256),0.5792754710746078)
((entropy,30,300),0.570642258679683)
((entropy,10,256),0.5678006650465051)
((entropy,20,300),0.5645890274211204)
((entropy,10,300),0.5548353562404907)

*/
```
可以看到，结果反而比之前差了很多。这说明这些特征的类别取值有倾斜。

##随机森林
随机森林可以理解将数据集合分成n个子集，然后在每个子集上建立决策树，最后结果是n棵决策树的平均值。
我们看一下所需要的参数：
```shell
scala> RandomForest.trainClassifier
<console>:42: error: ambiguous reference to overloaded definition,
both method trainClassifier in object RandomForest of type (input: org.apache.spark.api.java.JavaRDD[org.apache.spark.mllib.regression.LabeledPoint], numClasses: Int, categoricalFeaturesInfo: java.util.Map[Integer,Integer], numTrees: Int, featureSubsetStrategy: String, impurity: String, maxDepth: Int, maxBins: Int, seed: Int)org.apache.spark.mllib.tree.model.RandomForestModel
and  method trainClassifier in object RandomForest of type (input: org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint], numClasses: Int, categoricalFeaturesInfo: Map[Int,Int], numTrees: Int, featureSubsetStrategy: String, impurity: String, maxDepth: Int, maxBins: Int, seed: Int)org.apache.spark.mllib.tree.model.RandomForestModel
match expected type ?
       RandomForest.trainClassifier
                    ^

```
这里新增的参数有：

- numTrees：树的数量
- featureSubsetStrategy：我们看下[spark文档](http://spark.apache.org/docs/latest/mllib-ensembles.html#usage-tips):

 >featureSubsetStrategy: Number of features to use as candidates for splitting at each tree node. The number is specified as a fraction or function of the total number of features. Decreasing this number will speed up training, but can sometimes impact performance if too low.
 
我们可以将featureSubsetStrategy设置为auto，让算法自己来决定。
```scala
val forest = RandomForest.trainClassifier(
trainData, 7, Map[Int,Int](), 20, "auto", "entropy", 30, 300)
val predictionsAndLabels = data.map(example =>
 (forest.predict(example.features), example.label)
val mul = new  MulticlassMetrics(predictionsAndLabels)
mul.precision
//res59: Double = 0.8690027056239802    
```
