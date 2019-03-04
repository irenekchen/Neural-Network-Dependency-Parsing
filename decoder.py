from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def takeSecond(self, elem):
        return elem[1]

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            #ass
            # TODO: Write the body of this loop for part 4 
            
            features = self.extractor.get_input_representation(words, pos, state)
            features = features.reshape(1, -1)

            action_probabilities = self.model.predict(features)
            action_probabilities = [float(x) for x in action_probabilities[0]]
            indexed_action_probabilities = list(enumerate(action_probabilities))
            sorted_action_probabilities = sorted(indexed_action_probabilities, key=self.takeSecond, reverse=True)

            action_found = False
            count = 0
            while not action_found:
                index = sorted_action_probabilities[count][0]
                action = self.output_labels[index]
                if action[0] == "shift" and not (len(state.buffer) == 1 and not len(state.stack) == 0):
                    state.shift()
                    action_found = True
                elif action[0] == "right_arc" and not len(state.stack) == 0:
                    state.right_arc(action[1])
                    action_found = True
                elif action[0] == "left_arc" and not len(state.stack) == 0:
                    state.left_arc(action[1])
                    action_found = True
                else:
                    count += 1

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
