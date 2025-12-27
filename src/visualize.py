import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. Load the trained pipeline
try:
    pipeline = joblib.load("out/sentiment_model.pkl")
    print("Model loaded.")
except FileNotFoundError:
    print("Error: 'sentiment_model.pkl' not found. Run train_model.py first.")
    exit()

# Extract the fitted steps to get info needed for plotting
vectorizer_step = pipeline.named_steps['tfidfvectorizer']
classifier_step = pipeline.named_steps['decisiontreeclassifier']

# Get the actual words/bigrams the model learned
feature_names = vectorizer_step.get_feature_names_out()
# Get the class names corresponding to the model's internal ordering
class_names = classifier_step.classes_ # usually ['NEGATIVE', 'POSITIVE']

# Setup the canvas size (make it wide)
plt.figure(figsize=(25, 12))

# Plot the tree
plot_tree(classifier_step, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,      # Colors the boxes based on the majority class
          rounded=True,     # Makes boxes look nicer
          fontsize=12,      # Text size
          precision=2,  # Decimal places in the boxes
          max_depth=4)

plt.title("Decision Tree Flowchart for Sentiment Analysis")
output_file = "out/tree_diagram.png"
plt.savefig(output_file, dpi=100)
print(f"Success! Diagram saved as '{output_file}'")
# plt.show() # Uncomment this if you want it to pop up on your screen

from graphviz import Digraph

def draw_sentiment_fsm():
    # Initialize the specific "Finite Automaton" diagram type
    fsm = Digraph('Sentiment_FSM', format='svg')
    fsm.attr(rankdir='LR')  # Left to Right orientation
    
    # --- 1. Define States (Nodes) ---
    # Double circle for 'Final' states where we have a sentiment conclusion
    fsm.attr('node', shape='circle')
    fsm.node('START', 'Start\n(Neutral)', shape='doublecircle')
    
    fsm.node('POS', 'Positive\nState', shape='doublecircle', color='green')
    fsm.node('NEG', 'Negative\nState', shape='doublecircle', color='red')
    
    # The critical "Intermediate" state for handling "not..."
    fsm.node('INV', 'Inverter\n(Saw "not")', shape='circle', color='orange')

    # --- 2. Define Transitions (Edges) ---
    
    # From START
    fsm.edge('START', 'POS', label=' saw "good", "happy" ')
    fsm.edge('START', 'NEG', label=' saw "bad", "terrible" ')
    fsm.edge('START', 'INV', label=' saw "not", "no" ')
    fsm.edge('START', 'START', label=' saw neutral word ')

    # From POSITIVE (If we see more words, we might stay or switch)
    fsm.edge('POS', 'POS', label=' saw more pos words ')
    fsm.edge('POS', 'INV', label=' saw "not" ')
    
    # From NEGATIVE
    fsm.edge('NEG', 'NEG', label=' saw more neg words ')
    fsm.edge('NEG', 'INV', label=' saw "not" ')

    # --- THE CRITICAL LOGIC (Negation Handling) ---
    # This is where "not happy" becomes Negative
    fsm.edge('INV', 'NEG', label=' saw "good" (flipped!) ', color='red', fontcolor='red')
    
    # This is where "not bad" becomes Positive
    fsm.edge('INV', 'POS', label=' saw "bad" (flipped!) ', color='green', fontcolor='green')
    
    # Reset if neutral words appear after "not" (e.g. "not really...")
    fsm.edge('INV', 'INV', label=' neutral word ')

    # --- 3. Render ---
    output_name = 'out/sentiment_fsm_diagram'
    fsm.render(output_name, view=False)
    print(f"FSM Diagram generated: {output_name}.svg")

if __name__ == "__main__":
    draw_sentiment_fsm()