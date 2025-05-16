import os
import sys
import time
import argparse
import yaml
import csv
import pprint
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow_addons import optimizers
from kgcnn.utils.devices import set_devices_gpu
from kgcnn.data.crystal import CrystalDataset
from kgcnn.training.schedule import LinearWarmupExponentialDecay
from kgcnn.training.scheduler import LinearLearningRateScheduler
import kgcnn.training.callbacks
from models_tf.dimenetpp import load_dimenetpp_data, make_crystal_model

min_val = 1e-6

# 定义负对数似然损失函数
def nig_nll(y, mu, v, alpha, beta):
    """Calculate the negative log likelihood for a Normal-Inverse-Gamma distribution."""
    two_blambda = 2 * beta * (1 + v)
    pi = tf.constant(np.pi, dtype=y.dtype)
    nll = 0.5 * tf.math.log(pi / v) \
          - alpha * tf.math.log(two_blambda) \
          + (alpha + 0.5) * tf.math.log(v * (y - mu) ** 2 + two_blambda) \
          + tf.math.lgamma(alpha) \
          - tf.math.lgamma(alpha + 0.5)
    return nll

# 定义正则化损失函数
def nig_reg(y, mu, v, alpha):
    """Calculate the regularization term for evidential regression."""
    error = tf.abs(y - mu)
    evi = 2 * v + alpha
    return error * evi

# 定义证据回归损失函数
def evidential_regresssion_loss(y, pred):
    coeff = 0.01
    mu, logv, logalpha, logbeta = tf.split(pred, num_or_size_splits=4, axis=-1)

    # Squeeze and apply transformations
    mu = tf.squeeze(mu, axis=-1)
    logv = tf.squeeze(logv, axis=-1)
    logalpha = tf.squeeze(logalpha, axis=-1)
    logbeta = tf.squeeze(logbeta, axis=-1)

    logv = tf.nn.softplus(logv) + min_val
    logalpha = tf.nn.softplus(logalpha) + min_val + 1
    logbeta = tf.nn.softplus(logbeta) + min_val
    """Compute the total evidential regression loss combining NLL and regularization."""
    mu = tf.reshape(mu, [-1])
    v = tf.reshape(logv, [-1])
    alpha = tf.reshape(logalpha, [-1])
    beta = tf.reshape(logbeta, [-1])
    _y = tf.reshape(y, [-1])

    loss_nll = nig_nll(_y, mu, v, alpha, beta)
    loss_reg = nig_reg(_y, mu, v, alpha)
    return tf.reduce_mean(loss_nll) + (coeff * tf.reduce_mean(loss_reg))

def validate(loader, targets, ids, model, scaler, save_dir):
    predictions = model.predict(loader)
    predictions = tf.convert_to_tensor(predictions, dtype=tf.float32)
    # Split into components
    mu, logv, logalpha, logbeta = tf.split(predictions, num_or_size_splits=4, axis=-1)

    # Squeeze and apply transformations
    mu = tf.squeeze(mu, axis=-1)
    logv = tf.squeeze(logv, axis=-1)
    logalpha = tf.squeeze(logalpha, axis=-1)
    logbeta = tf.squeeze(logbeta, axis=-1)

    logv = tf.nn.softplus(logv) + min_val
    logalpha = tf.nn.softplus(logalpha) + min_val + 1
    logbeta = tf.nn.softplus(logbeta) + min_val

    # Flatten the tensors
    mu_flat = tf.reshape(mu, [-1])
    v_flat = tf.reshape(logv, [-1])
    alpha_flat = tf.reshape(logalpha, [-1])
    beta_flat = tf.reshape(logbeta, [-1])

    # Compute variance-like term
    var = beta_flat / (v_flat * (alpha_flat - 1))

    # Compute uncertainties directly using tensors
    epi_uncert = var
    ale_uncert = var * v_flat
    uncert = var + var * v_flat
    # Output is the flattened mu tensor
    output = mu_flat

    prediction, mc_uncert, _uncert, _epi_uncert, _ale_uncert = mc_dropout_predict(loader, model, n_dropout=50)

    def _mae(pred, tgt):
        """
        Computes the mean absolute error between prediction and target

        Parameters
        ----------

        pred: tf.Tensor (N, 1)
        tgt: tf.Tensor (N, 1)
        """
        return tf.reduce_mean(tf.abs(tgt - pred))

    # 计算MAE和有MC_Dropout的MAE
    output = scaler.inverse_transform(output.numpy().reshape(-1, 1))
    output = tf.convert_to_tensor(output, dtype=tf.float32)
    mae_errors = _mae(output, targets)
    prediction = scaler.inverse_transform(prediction.numpy().reshape(-1, 1))
    prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)
    mc_mae_errors = _mae(prediction, targets)

    # 更新不确定性
    uncert_errors = tf.reduce_mean(uncert)
    epi_uncert_errors = tf.reduce_mean(epi_uncert)
    ale_uncert_errors = tf.reduce_mean(ale_uncert)
    mc_uncert_errors = tf.reduce_mean(mc_uncert)
    mc_der_uncert_errors = tf.reduce_mean(_uncert)
    mc_der_uncert_e_errors = tf.reduce_mean(_epi_uncert)
    mc_der_uncert_a_errors = tf.reduce_mean(_ale_uncert)

    test_pred = output
    test_prediction = prediction
    test_target = targets
    test_uncert = uncert
    test_epi_uncert = epi_uncert
    test_ale_uncert = ale_uncert
    test_mc_uncert = mc_uncert
    test_mc_der_uncert = _uncert
    test_mc_der_uncert_a = _ale_uncert
    test_mc_der_uncert_e = _epi_uncert

    test_preds = tf.reshape(test_pred, [-1]).numpy().tolist()
    test_mc_preds = tf.reshape(test_prediction, [-1]).numpy().tolist()
    test_targets = tf.reshape(test_target, [-1]).numpy().tolist()
    test_cif_ids = ids
    test_uncerts = tf.reshape(test_uncert, [-1]).numpy().tolist()
    test_mc_uncerts = tf.reshape(test_mc_uncert, [-1]).numpy().tolist()
    test_mc_der_uncerts = tf.reshape(test_mc_der_uncert, [-1]).numpy().tolist()

    test_epi_uncerts = tf.reshape(test_epi_uncert, [-1]).numpy().tolist()
    test_ale_uncerts = tf.reshape(test_ale_uncert, [-1]).numpy().tolist()
    test_mc_der_uncerts_e = tf.reshape(test_mc_der_uncert_e, [-1]).numpy().tolist()
    test_mc_der_uncerts_a = tf.reshape(test_mc_der_uncert_a, [-1]).numpy().tolist()


    star_label = '**'
    with open(save_dir + '/' + 'test_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('test_cif_ids', 'test_targets', 'test_preds', 'test_mc_preds', 'test_mc_uncerts',
                         'test_uncerts', 'test_mc_der_uncerts', 'test_epi_uncerts', 'test_ale_uncerts',
                         'test_mc_der_uncerts_e', 'test_mc_der_uncerts_a'))
        for cif_id, target, pred, mc_pred, mc_uncertainty, uncertainty, mc_der_uncertainty, epi_uncertainty, ale_uncertainty, mc_der_uncerts_e, mc_der_uncerts_a in zip(
                test_cif_ids, test_targets,
                test_preds, test_mc_preds, test_mc_uncerts, test_uncerts, test_mc_der_uncerts,
                test_epi_uncerts, test_ale_uncerts, test_mc_der_uncerts_e, test_mc_der_uncerts_a):
            writer.writerow((cif_id, target, pred, mc_pred, mc_uncertainty, uncertainty, mc_der_uncertainty,
                             epi_uncertainty, ale_uncertainty, mc_der_uncerts_e, mc_der_uncerts_a))

    print(' {star} MAE {mae_errors:.3f}'.format(star=star_label,
                                                mae_errors=mae_errors))
    print(' {star} MC_MAE {mc_mae_errors:.3f}'.format(star=star_label,
                                                      mc_mae_errors=mc_mae_errors))
    print(
        ' {star} Uncertainty {uncert_errors:.3f}, epi_Uncertainty {epi_uncert_errors:.3f}, ale_Uncertainty {ale_uncert_errors:.3f}'.format(
            star=star_label,
            uncert_errors=uncert_errors, epi_uncert_errors=epi_uncert_errors,
            ale_uncert_errors=ale_uncert_errors))
    print(
        ' {star} MC_Uncertainty {mc_uncert_errors:.3f}, MC_DER_Uncertainty {mc_der_uncert_errors:.3f}, MC_E_Uncertainty {mc_der_uncert_e_errors:.3f}, MC_A_Uncertainty {mc_der_uncert_a_errors:.3f}'.format(
            star=star_label,
            mc_uncert_errors=mc_uncert_errors, mc_der_uncert_errors=mc_der_uncert_errors,
            mc_der_uncert_e_errors=mc_der_uncert_e_errors, mc_der_uncert_a_errors=mc_der_uncert_a_errors))

    return (
        mae_errors, mc_mae_errors, mc_uncert_errors, uncert_errors, epi_uncert_errors,
        ale_uncert_errors, mc_der_uncert_errors, mc_der_uncert_e_errors, mc_der_uncert_a_errors
    )

def mc_dropout_predict(loader, model, n_dropout):
    # 启用dropout
    predictions = []
    for t in range(n_dropout):
        pred = model(loader, training=True)
        predictions.append(pred)
    _mu = []
    _v = []
    _alpha = []
    _beta = []
    for n in range(n_dropout):
        # Split into components
        mu, logv, logalpha, logbeta = tf.split(predictions[n], num_or_size_splits=4, axis=-1)

        # Squeeze and apply transformations
        mu = tf.squeeze(mu, axis=-1)
        logv = tf.squeeze(logv, axis=-1)
        logalpha = tf.squeeze(logalpha, axis=-1)
        logbeta = tf.squeeze(logbeta, axis=-1)

        logv = tf.nn.softplus(logv) + min_val
        logalpha = tf.nn.softplus(logalpha) + min_val + 1
        logbeta = tf.nn.softplus(logbeta) + min_val

        _mu.append(tf.reshape(mu, [-1]))
        _v.append(tf.reshape(logv, [-1]))
        _alpha.append(tf.reshape(logalpha, [-1]))
        _beta.append(tf.reshape(logbeta, [-1]))

    # Stack and compute mean/variance
    prediction = tf.reduce_mean(tf.stack(_mu, axis=0), axis=0)
    mc_uncert = tf.math.reduce_variance(tf.stack(_mu, axis=0), axis=0)
    _v = tf.reduce_mean(tf.stack(_v, axis=0), axis=0)
    _alpha = tf.reduce_mean(tf.stack(_alpha, axis=0), axis=0)
    _beta = tf.reduce_mean(tf.stack(_beta, axis=0), axis=0)

    prediction = tf.reshape(prediction, [-1])
    mc_uncert = tf.reshape(mc_uncert, [-1])
    _v = tf.reshape(_v, [-1])
    _alpha = tf.reshape(_alpha, [-1])
    _beta = tf.reshape(_beta, [-1])
    _var = _beta / (_v * (_alpha - 1))
    _epi_uncert = _var
    _ale_uncert = _var*_v
    _uncert = _var + _var * _v

    return prediction, mc_uncert, _uncert, _epi_uncert, _ale_uncert


if __name__ == '__main__':
    start_time = time.time()
    print("Start time:", start_time)

    parser = argparse.ArgumentParser(description="OOD MatBench with UQ")
    parser.add_argument("--task", default="perovskites", type=str, help="dielectric, elasticity, perovskites, jdft2d, supercon3d, mp_gap")
    parser.add_argument("--data_path", default="./data", type=str, help="path to data")
    parser.add_argument("--config_path", default="config.yml", type=str, help="path to config file")
    parser.add_argument("--model", default="DimeNetPP_TF", type=str, help="DimeNetPP_TF")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("--epochs", default=None, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", default=None, type=float, help="initial learning rate")
    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--oods", default="LOCO", type=str, help="LOCO, SparseXcluster, SparseYcluster, SparseXsingle, SparseYsingle")

    args = parser.parse_args(sys.argv[1:])

    assert os.path.exists(args.config_path), (
            "Config file not found in " + args.config_path
    )
    with open(args.config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    for key in config["Models"]:
        if args.epochs != None:
            config["Models"][key]["epochs"] = args.epochs
        if args.batch_size != None:
            config["Models"][key]["batch_size"] = args.batch_size
        if args.lr != None:
            config["Models"][key]["lr"] = args.lr

    if args.model != None:
        config["Models"] = config["Models"].get(args.model)

    print("Settings: ")
    pprint.pprint(config)
    with open(args.task + "_" + args.model + "_settings.txt", "w") as log_file:
        pprint.pprint(config, log_file)

    set_devices_gpu(0)

    data_process_start_time = time.time()

    if config["Models"]["model"]=="DimeNetPP_TF":
        cif_ids, targets, crystals = load_dimenetpp_data(args.data_path, args.task)
        scaler = StandardScaler(with_std=True, with_mean=True, copy=True)
        tensor_np = tf.convert_to_tensor(targets)
        tensor_np = tensor_np.numpy()
        tensor_2d = tensor_np.reshape(-1, 1) if tensor_np.ndim == 1 else tensor_np
        scaler.fit(tensor_2d)

    #oods = ["LOCO", "SparseXcluster", "SparseYcluster", "SparseXsingle", "SparseYsingle"]
    oods = [args.oods]
    task_mae = {}
    task_uncert = {}
    task_uncert_E = {}
    task_uncert_A = {}
    task_mae_mc = {}
    task_uncert_mc = {}
    task_uncert_mc_der = {}
    task_uncert_mc_E = {}
    task_uncert_mc_A = {}
    for ood in oods:
        train_sets_file = "folds/" + args.task + "_folds/train/OFM_" + args.task + "_" + ood + "_target_clusters50_train.json"
        valid_sets_file = "folds/" + args.task + "_folds/val/OFM_" + args.task + "_" + ood + "_target_clusters50_val.json"
        test_sets_file = "folds/" + args.task + "_folds/test/OFM_" + args.task + "_" + ood + "_target_clusters50_test.json"
        assert os.path.exists(train_sets_file), (
            "OOD train set indexs file not found"
        )
        assert os.path.exists(valid_sets_file), (
            "OOD val set indexs file not found"
        )
        assert os.path.exists(test_sets_file), (
            "OOD test set indexs file not found"
        )
        with open(train_sets_file, 'r', encoding='utf-8') as fileT:
            train_indexs = json.load(fileT)
        with open(valid_sets_file, 'r', encoding='utf-8') as fileV:
            val_indexs = json.load(fileV)
        with open(test_sets_file, 'r', encoding='utf-8') as fileT:
            test_indexs = json.load(fileT)

        mae_errors=[]
        uncert_errors=[]
        epi_uncert_errors=[]
        ale_uncert_errors=[]
        mc_mae_errors=[]
        mc_uncert_errors=[]
        mc_der_uncert_errors=[]
        mc_der_uncert_a_errors=[]
        mc_der_uncert_e_errors=[]

        for i in range(1,2):
            train_index = train_indexs[str(i)]
            val_index = val_indexs[str(i)]
            test_index = test_indexs[str(i)]

            if config["Models"]["model"] == "DimeNetPP_TF":
                callbacks = {
                    "graph_labels": lambda st, ds: np.expand_dims(ds, axis=-1),
                    "node_coordinates": lambda st, ds: np.array(st.cart_coords, dtype="float"),
                    "node_frac_coordinates": lambda st, ds: np.array(st.frac_coords, dtype="float"),
                    "graph_lattice": lambda st, ds: np.ascontiguousarray(np.array(st.lattice.matrix), dtype="float"),
                    "abc": lambda st, ds: np.array(st.lattice.abc),
                    "charge": lambda st, ds: np.array([st.charge], dtype="float"),
                    "volume": lambda st, ds: np.array([st.lattice.volume], dtype="float"),
                    "node_number": lambda st, ds: np.array(st.atomic_numbers, dtype="int"),
                }
                methods = [
                    {"map_list": {"method": "set_range_periodic", "max_distance": 8.0, "max_neighbours": 17}},
                    {"map_list": {"method": "set_angle", "allow_multi_edges": True}}
                ]
                inputs = [{"shape": [None], "name": "node_number", "dtype": "float32", "ragged": True},
                           {"shape": [None, 3], "name": "node_coordinates", "dtype": "float32", "ragged": True},
                           {"shape": [None, 2], "name": "range_indices", "dtype": "int64", "ragged": True},
                           {"shape": [None, 2], "name": "angle_indices", "dtype": "int64", "ragged": True},
                           {'shape': (None, 3), 'name': "range_image", 'dtype': 'int64', 'ragged': True},
                           {'shape': (3, 3), 'name': "graph_lattice", 'dtype': 'float32', 'ragged': False}
                           ]
                input_embedding = {"node": {"input_dim": 95, "output_dim": 128,
                                             "embeddings_initializer": {"class_name": "RandomUniform",
                                                                        "config": {"minval": -1.7320508075688772,
                                                                                   "maxval": 1.7320508075688772}}}}

                # train data
                cif_to_index = {str(value): idx for idx, value in enumerate(cif_ids)}
                train_inputs = [crystals[cif_to_index[str(i)]] for i in train_index if str(i) in cif_to_index]
                train_outputs = [targets[cif_to_index[str(i)]] for i in train_index if str(i) in cif_to_index]
                train_ids = [i for i in train_index if str(i) in cif_to_index]
                data_train = CrystalDataset()
                data_train._map_callbacks(train_inputs, pd.Series(train_outputs), callbacks)
                print("Making graph... (this may take a while)")
                data_train.set_methods(methods)
                data_train.clean(inputs)
                y_train = np.array(data_train.get("graph_labels"))
                x_train = data_train.tensor(inputs)
                y_train = scaler.transform(y_train.reshape(-1, 1))
                y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
                print("Train data shape:", y_train.shape)

                # val data
                val_inputs = [crystals[cif_to_index[str(i)]] for i in val_index if str(i) in cif_to_index]
                val_outputs = [targets[cif_to_index[str(i)]] for i in val_index if str(i) in cif_to_index]
                val_ids = [i for i in val_index if str(i) in cif_to_index]
                data_val = CrystalDataset()
                data_val._map_callbacks(val_inputs, pd.Series(val_outputs), callbacks)
                print("Making graph... (this may take a while)")
                data_val.set_methods(methods)
                data_val.clean(inputs)
                y_val = np.array(data_val.get("graph_labels"))
                x_val = data_val.tensor(inputs)
                y_val = scaler.transform(y_val.reshape(-1, 1))
                y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
                print("Validation data shape:", y_val.shape)

                # train and validate your model
                model = make_crystal_model(**(config["Models"]["model_setting"]), extensive=False, use_output_mlp=False, output_mlp={},
                                           inputs=inputs, input_embedding=input_embedding)

                # Get trainable parameters count
                trainable_params = sum(tf.size(var).numpy() for var in model.trainable_variables)
                print(f'Number of M parameters = {trainable_params:,}')

                optimizer = {
                    "class_name": "Addons>MovingAverage",
                    "config": {
                        "optimizer": {
                            "class_name": "Adam",
                            "config": {
                                "learning_rate": {
                                    "class_name": "kgcnn>LinearWarmupExponentialDecay",
                                    "config": {
                                        "learning_rate": 0.001, "warmup_steps": 3000.0, "decay_steps": 4000000.0,
                                        "decay_rate": 0.01
                                    }
                                }, "amsgrad": True
                            }
                        },
                        "average_decay": 0.999
                    }
                }
                # optimizer = {
                #     "class_name": "Addons>MovingAverage",
                #     "config": {
                #         "optimizer": {
                #             "class_name": "Adam",
                #             "config": {
                #                 "learning_rate": {
                #                     "class_name": "kgcnn>LinearWarmupExponentialDecay",
                #                     "config": {
                #                         "learning_rate": 0.001, "warmup_steps": 10.0, "decay_steps": 10.0,
                #                         "decay_rate": 0.01
                #                     }
                #                 }, "amsgrad": True
                #             }
                #         },
                #         "average_decay": 0.999
                #     }
                # }

                model.compile(
                    loss=evidential_regresssion_loss,
                    optimizer=tf.keras.optimizers.get(optimizer)
                )


                save_dir = 'results/' + args.model + '/' + args.task + '/' + ood + '/fold_' + str(i)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                checkpoint_path = os.path.join(save_dir, 'model_best.h5')

                tf_callbacks = [
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_path,
                        monitor='val_loss',  # Monitor validation loss (deep evidence regression loss)
                        save_best_only=True,
                        save_weights_only=True,
                        verbose=1
                    ),
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=50,  # Stop if no improvement after 50 epochs
                        verbose=1,
                        restore_best_weights=False  # Best weights saved by ModelCheckpoint
                    )
                ]

                # Train the model with validation
                hist = model.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val), # Use validation set during training
                    batch_size=config["Models"]["batch_size"],
                    epochs=config["Models"]["epochs"],
                    callbacks=tf_callbacks
                )

                # Test the best model
                # print('---------Evaluate Model on Test Set---------------')
                # model.load_weights(checkpoint_path)  # Load best weights
                # # test data
                # test_inputs = [crystals[cif_to_index[str(i)]] for i in test_index if str(i) in cif_to_index]
                # test_outputs = [targets[cif_to_index[str(i)]] for i in test_index if str(i) in cif_to_index]
                # test_ids = [i for i in test_index if str(i) in cif_to_index]
                # data_test = CrystalDataset()
                # data_test._map_callbacks(test_inputs, pd.Series(np.zeros(len(test_inputs))), callbacks)
                # print("Making graph... (this may take a while)")
                # data_test.set_methods(methods)
                # #removed = data_test.clean(inputs)
                # x_test = data_test.tensor(inputs)
                # y_test = tf.convert_to_tensor(test_outputs, dtype=tf.float32)
                # #indices_test = [j for j in range(len(test_inputs))]
                # #for j in removed:
                # #    indices_test.pop(j)
                # #predictions = np.expand_dims(np.zeros(len(test_inputs), dtype="float"), axis=-1)
                # #predictions[np.array(indices_test)] = predictions_model
                # (mae_error, mc_mae_error, mc_uncert_error, uncert_error, epi_uncert_error, ale_uncert_error,
                #  mc_der_uncert_error, mc_der_uncert_e_error, mc_der_uncert_a_error) = validate(x_test, y_test, test_ids,
                #                                                                                model, scaler, save_dir)

            # mae_errors.append(mae_error)
            # uncert_errors.append(uncert_error)
            # epi_uncert_errors.append(epi_uncert_error)
            # ale_uncert_errors.append(ale_uncert_error)
            #
            # mc_mae_errors.append(mc_mae_error)
            # mc_uncert_errors.append(mc_uncert_error)
            # mc_der_uncert_errors.append(mc_der_uncert_error)
            # mc_der_uncert_a_errors.append(mc_der_uncert_a_error)
            # mc_der_uncert_e_errors.append(mc_der_uncert_e_error)

    #     mae_errors = tf.reshape(tf.stack(mae_errors), [-1]).numpy().tolist()
    #     uncert_errors = tf.reshape(tf.stack(uncert_errors), [-1]).numpy().tolist()
    #     epi_uncert_errors = tf.reshape(tf.stack(epi_uncert_errors), [-1]).numpy().tolist()
    #     ale_uncert_errors = tf.reshape(tf.stack(ale_uncert_errors), [-1]).numpy().tolist()
    #     mc_mae_errors = tf.reshape(tf.stack(mc_mae_errors), [-1]).numpy().tolist()
    #     mc_uncert_errors = tf.reshape(tf.stack(mc_uncert_errors), [-1]).numpy().tolist()
    #     mc_der_uncert_errors = tf.reshape(tf.stack(mc_der_uncert_errors), [-1]).numpy().tolist()
    #     mc_der_uncert_a_errors = tf.reshape(tf.stack(mc_der_uncert_a_errors), [-1]).numpy().tolist()
    #     mc_der_uncert_e_errors = tf.reshape(tf.stack(mc_der_uncert_e_errors), [-1]).numpy().tolist()
    #
    #
    #     mae = np.array(mae_errors, dtype=float).mean()
    #     mae_std = np.array(mae_errors, dtype=float).std()
    #     uncertainty = np.array(uncert_errors, dtype=float).mean()
    #     uncertainty_std = np.array(uncert_errors, dtype=float).std()
    #     epi_uncertainty = np.array(epi_uncert_errors, dtype=float).mean()
    #     epi_uncertainty_std = np.array(epi_uncert_errors, dtype=float).std()
    #     ale_uncertainty = np.array(ale_uncert_errors, dtype=float).mean()
    #     ale_uncertainty_std = np.array(ale_uncert_errors, dtype=float).std()
    #     mc_mae = np.array(mc_mae_errors, dtype=float).mean()
    #     mc_mae_std = np.array(mc_mae_errors, dtype=float).std()
    #     mc_uncertainty = np.array(mc_uncert_errors, dtype=float).mean()
    #     mc_uncertainty_std = np.array(mc_uncert_errors, dtype=float).std()
    #     mc_der_uncertainty = np.array(mc_der_uncert_errors, dtype=float).mean()
    #     mc_der_uncertainty_std = np.array(mc_der_uncert_errors, dtype=float).std()
    #     mc_e_uncertainty = np.array(mc_der_uncert_e_errors, dtype=float).mean()
    #     mc_e_uncertainty_std = np.array(mc_der_uncert_e_errors, dtype=float).std()
    #     mc_a_uncertainty = np.array(mc_der_uncert_a_errors, dtype=float).mean()
    #     mc_a_uncertainty_std = np.array(mc_der_uncert_a_errors, dtype=float).std()
    #
    #     with open('results/' + args.model + '/' + args.task + '/' + ood + '/' + 'test_metrics.csv', 'w') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(('mae', 'uncertainty', 'epi_uncertainty', 'ale_uncertainty', 'mc_mae', 'mc_uncertainty',
    #                          'mc_der_uncertainty', 'mc_e_uncertainty', 'mc_a_uncertainty'))
    #         for mae_err, uncert_err, epi_uncert_err, ale_uncert_err, mc_mae_err, mc_uncert_err, mc_der_uncert_err, mc_der_uncert_e_err, mc_der_uncert_a_err in zip(
    #                 mae_errors, uncert_errors, epi_uncert_errors,
    #                 ale_uncert_errors, mc_mae_errors, mc_uncert_errors, mc_der_uncert_errors,
    #                 mc_der_uncert_e_errors, mc_der_uncert_a_errors):
    #             writer.writerow((mae_err, uncert_err, epi_uncert_err, ale_uncert_err, mc_mae_err, mc_uncert_err,
    #                              mc_der_uncert_err, mc_der_uncert_e_err, mc_der_uncert_a_err))
    #         writer.writerow(('mae(std)', 'uncertainty(std)', 'epi(std)', 'ale(std)', 'mc_mae(std)',
    #                          'mc_uncertainty(std)', 'mc_der_uncertainty(std)', 'mc_e(std)', 'mc_a(std)'))
    #         writer.writerow((f'{mae}({mae_std})', f'{uncertainty}({uncertainty_std})',
    #                          f'{epi_uncertainty}({epi_uncertainty_std})',
    #                          f'{ale_uncertainty}({ale_uncertainty_std})',
    #                          f'{mc_mae}({mc_mae_std})', f'{mc_uncertainty}({mc_uncertainty_std})',
    #                          f'{mc_der_uncertainty}({mc_der_uncertainty_std})',
    #                          f'{mc_e_uncertainty}({mc_e_uncertainty_std})',
    #                          f'{mc_a_uncertainty}({mc_a_uncertainty_std})'))
    #
    #     task_mae[ood] = (mae, mae_std)
    #     task_uncert[ood] = (uncertainty, uncertainty_std)
    #     task_uncert_E[ood] = (epi_uncertainty, epi_uncertainty_std)
    #     task_uncert_A[ood] = (ale_uncertainty, ale_uncertainty_std)
    #     task_mae_mc[ood] = (mc_mae, mc_mae_std)
    #     task_uncert_mc[ood] = (mc_uncertainty, mc_uncertainty_std)
    #     task_uncert_mc_der[ood] = (mc_der_uncertainty, mc_der_uncertainty_std)
    #     task_uncert_mc_E[ood] = (mc_e_uncertainty, mc_e_uncertainty_std)
    #     task_uncert_mc_A[ood] = (mc_a_uncertainty, mc_a_uncertainty_std)
    #
    # print(args.task + '_mae：', task_mae)
    # print(args.task + '_uncert：', task_uncert)
    # print(args.task + '_uncert_E:', task_uncert_E)
    # print(args.task + '_uncert_A:', task_uncert_A)
    # print(args.task + '_mae_mc：', task_mae_mc)
    # print(args.task + '_uncert_mc：', task_uncert_mc)
    # print(args.task + '_uncert_mc_der：', task_uncert_mc_der)
    # print(args.task + '_uncert_mc_E:', task_uncert_mc_E)
    # print(args.task + '_uncert_mc_A:', task_uncert_mc_A)
    # print("--- %s seconds for the entire experiment time  ---" % (time.time() - start_time))
