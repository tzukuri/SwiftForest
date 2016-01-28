import Foundation

// ---------------------------------------
// classification
// ---------------------------------------
public class Forest: Classifier {
    public var trees: [Tree]
    public var model: Model
    
    public init(model: Model, trees: [Tree]) {
        self.model = model
        self.trees = trees
    }

    public func classify(values: [Double]) -> Int {
        var counts = [Int: Int]()
        
        // classify values in each tree and count number of times
        // each output class is selected
        for tree in trees {
            let output = tree.classify(values)
            counts[output] = (counts[output] ?? 0) + 1
        }
        
        // sort output classes by count ($0 and $1 are (key, value))
        // maxElement expects result of: $0 ordered before $1
        if let mode = counts.maxElement({ $0.1 < $1.1 }) {
            return mode.0
        } else {
            return counts.values.first!
        }
    }
}



// ---------------------------------------
// training
// ---------------------------------------
final public class TrainableForest: Forest, TrainableClassifier {
    public var delegate: ClassifierDelegate?
    public var randomSeed: Int
    
    public init(
        model: Model,
        size: Int = 100,
        pickFeatures: Int? = nil,
        minExamples: Int = 2,
        randomSeed: Int = 1,
        delegate: ClassifierDelegate? = nil
    ) {
        self.randomSeed = randomSeed
        self.delegate = delegate
        var trees = [Tree]()

        for _ in 0..<size {
            trees.append(
                TrainableTree(model: model, pickFeatures: pickFeatures, minExamples: minExamples)
            )
        }

        super.init(model: model, trees: trees)
    }
    
    public func train(trainingSet: TrainingSet) {
        delegate?.trainingWillStartWithSteps(trees.count)
        srand48(randomSeed)

        // set the random seed for each tree before looping so the
        // call to lrand48 doesn't happen at different times when
        // using multiple threads
        for tree in trees as! [TrainableTree] {
            tree.seedRandomGenerator(Int32(lrand48()))
        }

        // use GCD to parallelise tree training
        let queue = dispatch_queue_create("com.tzu.dt.trees", DISPATCH_QUEUE_CONCURRENT)
        dispatch_apply(trees.count, queue) { index in
            let tree = self.trees[index] as! TrainableTree
            tree.train(trainingSet)
            self.delegate?.trainingDidCompleteStep()
        }

        delegate?.trainingDidFinish()
    }
}
