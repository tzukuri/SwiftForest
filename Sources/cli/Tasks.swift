import Foundation
import SwiftForest
import SwiftCLI
import Progress

// ---------------------------------------
// CSV file loading
// ---------------------------------------
struct LoadDataTaskDelegate {
    var before: () -> Void
    var after: (rowSet: RowSet, model: Model) -> Void
}

class LoadDataTask {
    var delegate: LoadDataTaskDelegate?

    // data
    var paths: [String]!
    var outputsPresent: Bool

    // options
    var removeFeatures: [Int]? = nil
    var shuffle = false

    init(outputsPresent: Bool, shuffle: Bool, delegate: LoadDataTaskDelegate?) {
        self.outputsPresent = outputsPresent
        self.shuffle = shuffle
        self.delegate = delegate
    }

    func run() -> (RowSet, Model) {
        var rowSet: RowSet!
        var model: Model!
        delegate?.before()

        // the first call to read will create the model and rowset objects.
        // further calls mutate the objects in place so rows don't need
        // to be copied and merged together.
        for path in paths {
            CSV.read(path, rowSet: &rowSet, model: &model, outputsPresent: outputsPresent)
        }

        if let indexes = removeFeatures {
            rowSet.removeFeatures(indexes)
            model.removeFeatures(indexes)
        }

        // convert string output classes to an output index
        if outputsPresent {
            let trainingSet = rowSet as! TrainingSet
            trainingSet.setOutputIndexes(model)
        }

        if shuffle {
            rowSet.shuffle()
        }

        delegate?.after(rowSet: rowSet, model: model)
        return (rowSet, model)
    }
}


// ---------------------------------------
// training/loading classifier
// ---------------------------------------
protocol ClassifierProvider {
    func run() -> Classifier
}

struct TrainingTaskDelegate: ClassifierDelegate {
    var before: (task: TrainingTask) -> Void
    var after: (classifier: Classifier, duration: Double) -> Void
    var bar: ProgressBar?

    init(before: (task: TrainingTask) -> Void, after: (classifier: Classifier, duration: Double) -> Void) {
        self.before = before
        self.after = after
    }

    mutating func progressSteps(count: Int) {
        bar = ProgressBar(count: count)
    }

    mutating func step() {
        bar?.next()
    }

    mutating func finish() {
        bar?.next()
    }
}

class TrainingTask: ClassifierProvider {
    var delegate: TrainingTaskDelegate?

    // data
    var loadDataTask: LoadDataTask!

    // options
    var pickFeatures: Int? = nil
    var forestSize = 100
    var minExamples = 2
    var randomSeed = 1

    init(delegate: TrainingTaskDelegate?) {
        self.delegate = delegate
    }

    func run() -> Classifier {
        // load data
        let (rowSet, model) = loadDataTask.run()
        let trainingSet = rowSet as! TrainingSet

        // prepare trainer
        let classifier = TrainableForest(
            model: model,
            size: forestSize,
            pickFeatures: pickFeatures,
            minExamples: minExamples,
            randomSeed: randomSeed,
            delegate: delegate
        )

        // train
        delegate?.before(task: self)
        let start = NSDate()
        classifier.train(trainingSet)
        let duration = start.timeIntervalSinceNow * -1

        delegate?.after(classifier: classifier, duration: duration)
        return classifier
    }
}

struct LoadClassifierTaskDelegate {
    var before: (task: LoadClassifierTask) -> Void
    var after: (classifier: Classifier, duration: Double) -> Void
}

class LoadClassifierTask: ClassifierProvider {
    var delegate: LoadClassifierTaskDelegate?
    var path: String!

    init(delegate: LoadClassifierTaskDelegate?) {
        self.delegate = delegate
    }

    func run() -> Classifier {
        delegate?.before(task: self)

        let start = NSDate()
        let classifier = SerialisedForest.read(path)
        let duration = start.timeIntervalSinceNow * -1

        delegate?.after(classifier: classifier, duration: duration)
        return classifier
    }
}


// ---------------------------------------
// classification
// ---------------------------------------
struct ClassificationTaskDelegate {
    var before: () -> Void
    var after: (duration: Double) -> Void
    var bar: ProgressBar?

    init(before: () -> Void, after: (duration: Double) -> Void) {
        self.before = before
        self.after = after
    }

    mutating func progressSteps(count: Int) {
        bar = ProgressBar(count: count)
    }

    mutating func step() {
        bar?.next()
    }

    mutating func finish() {
        bar?.next()
    }
}

class ClassificationTask {
    var delegate: ClassificationTaskDelegate?

    // forest/tree
    var classifierProvider: ClassifierProvider!

    // data
    var loadDataTask: LoadDataTask!
    var rowSet: RowSet!

    init(delegate: ClassificationTaskDelegate?) {
        self.delegate = delegate
    }

    func run() -> [Int] {
        // run prior tasks
        let classifier = classifierProvider.run()
        let (rows, _) = loadDataTask.run()
        rowSet = rows
        
        // classify
        delegate?.before()
        delegate?.progressSteps(rows.count)
        let start = NSDate()
        let classifications = rows.map {(row) -> Int in
            delegate?.step()
            return classifier.classify(row)
        }

        let duration = start.timeIntervalSinceNow * -1
        delegate?.finish()

        delegate?.after(duration: duration)
        return classifications
    }
}


// ---------------------------------------
// testing
// ---------------------------------------
class TestTask {
    var classificationTask: ClassificationTask!

    func run() -> Double {
        let classifications = classificationTask.run()
        let examples = classificationTask.rowSet as! TrainingSet
        var correct = 0.0

        for (example, classification) in zip(examples.examples, classifications) {
            if classification == example.output {
                correct += 1.0
            }
        }

        return correct / Double(examples.count)
    }
}

