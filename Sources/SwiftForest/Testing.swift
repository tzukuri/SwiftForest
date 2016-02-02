import Foundation

// ---------------------------------------
// testing
// ---------------------------------------
final public class TestSet {
    public var trainingSet: TrainingSet
    public var testExamples: ArraySlice<Example>

    public init(trainingSet: TrainingSet, testExamples: ArraySlice<Example>) {
        self.trainingSet = trainingSet
        self.testExamples = testExamples
    }

    public convenience init(trainingSet: TrainingSet, testRange: Range<Int>) {
        var trainingExamples = trainingSet.examples

        // extract test examples in the test fold range
        let testExamples = trainingExamples[testRange]

        // remove these from the training set
        trainingExamples.removeRange(testRange)

        let trainingSet = TrainingSet(examples: trainingExamples)
        self.init(trainingSet: trainingSet, testExamples: testExamples)
    }

    public func score(classifier: TrainableClassifier) -> Double {
        var correct = 0.0

        for example in testExamples {
            let output = classifier.classify(example.values)
            if output == example.output {
                correct += 1.0
            }
        }

        return correct / Double(testExamples.count)
    }

    public func train(classifier: TrainableClassifier) {
        classifier.train(trainingSet)
    }
}



// ---------------------------------------
// X-cross fold validation
// ---------------------------------------
public protocol CrossFoldValidationDelegate {
    mutating func validationProgress(fold: Int)
}

final public class CrossFoldValidation {
    public var delegate: CrossFoldValidationDelegate?
    public var trainingSet: TrainingSet
    public var testSets = [TestSet]()

    public init(trainingSet: TrainingSet, folds: Int, delegate: CrossFoldValidationDelegate?) {
        self.delegate = delegate
        self.trainingSet = trainingSet
        let foldSize = trainingSet.examples.count / folds

        for i in 0..<folds {
            let range = (i * foldSize)..<((i + 1) * foldSize)
            testSets.append(TestSet(trainingSet: trainingSet, testRange: range))
        }
    }

    public func score(classifier: TrainableClassifier, maxFolds: Int? = nil) -> Double {
        let folds = maxFolds ?? testSets.count
        var sum = 0.0

        for (i, testSet) in testSets.enumerate() {
            if i >= folds { break }
            delegate?.validationProgress(i)
            testSet.train(classifier)
            sum += testSet.score(classifier)
        }

        return sum / Double(folds)
    }
}
