import uff

def convert_to_uff():
   
    print('---------------convert to uff--------------------')

    graph_def = uff.from_tensorflow_frozen_model('model.pb')
    with open('model.uff','wb') as f:
        f.write(graph_def)
    print('---------------saved to model.uff--------------------')

convert_to_uff()
