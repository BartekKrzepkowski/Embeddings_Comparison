import os
import keras
import pydot as pyd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils.extmath import cartesian
from keras.utils import vis_utils as vizu



def display_model_history(history, model_name, ax):
    keys_nd = np.array(list(history.keys())).reshape(ax.shape[0], ax.shape[1])
    for i, j in cartesian((np.arange(ax.shape[0]), np.arange(ax.shape[1]))):
        key = keys_nd[i][j]
        ax[i][j].plot(history[key], label=model_name)
        ax[i][j].title.set_text(key)
        ax[i][j].set_xlabel("epoch")
        ax[i][j].set_ylabel("metric_value")
        ax[i][j].grid(True)
        ax[i][j].legend()

        
def display_results(history_dict, nrows, ncols, save_img=False, markered_path=None):
    names = []
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,10))
    for model_name in history_dict:
        names.append(model_name)
        display_model_history(history=history_dict[model_name], model_name=model_name, ax=ax)
           
    if save_img:
        if len(names) == 1:
            fig.savefig("{}/{}_history_figure.png".format(markered_path, names[0]), dpi=fig.dpi)
        else:
            fig.savefig("about_models/{}/{}/{}_history_figure.png".format("comparision", time_marker, str(names)), dpi=fig.dpi)
        

def save_report(model, model_name, models_history, save_model=False):
    keras.utils.vis_utils.pydot = pyd
    time_marker = datetime.now().strftime("%Y%m%d-%H%M%S")
    markered_path = "about_models/{}/{}".format(model_name, time_marker)
    
    if not os.path.exists(markered_path):
        os.makedirs(markered_path)
        
    display_results(history_dict={model_name: models_history}, nrows=2, ncols=len(models_history)//2, save_img=True, markered_path=markered_path)
    vizu.plot_model(model, "{}/{}_vizu.png".format(markered_path, model_name), show_layer_names=False, show_shapes=True)
    with open('{}/{}_report.txt'.format(markered_path, model_name), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    if save_model:
        model.save('{}/{}.h5'.format(markered_path, model_name))
    