mae_errors = []
uncert_errors = []
epi_uncert_errors = []
ale_uncert_errors = []
mc_mae_errors = []
mc_uncert_errors = []
mc_der_uncert_errors = []
mc_der_uncert_a_errors = []
mc_der_uncert_e_errors = []
for i in range(50):
    mae_errors.append(mae_error)
    uncert_errors.append(uncert_error)
    epi_uncert_errors.append(epi_uncert_error)
    ale_uncert_errors.append(ale_uncert_error)

    mc_mae_errors.append(mc_mae_error)
    mc_uncert_errors.append(mc_uncert_error)
    mc_der_uncert_errors.append(mc_der_uncert_error)
    mc_der_uncert_a_errors.append(mc_der_uncert_a_error)
    mc_der_uncert_e_errors.append(mc_der_uncert_e_error)
mae = np.array(mae_errors, dtype=float).mean()
mae_std = np.array(mae_errors, dtype=float).std()
uncertainty = np.array(uncert_errors, dtype=float).mean()
uncertainty_std = np.array(uncert_errors, dtype=float).std()
epi_uncertainty = np.array(epi_uncert_errors, dtype=float).mean()
epi_uncertainty_std = np.array(epi_uncert_errors, dtype=float).std()
ale_uncertainty = np.array(ale_uncert_errors, dtype=float).mean()
ale_uncertainty_std = np.array(ale_uncert_errors, dtype=float).std()
mc_mae = np.array(mc_mae_errors, dtype=float).mean()
mc_mae_std = np.array(mc_mae_errors, dtype=float).std()
mc_uncertainty = np.array(mc_uncert_errors, dtype=float).mean()
mc_uncertainty_std = np.array(mc_uncert_errors, dtype=float).std()
mc_der_uncertainty = np.array(mc_der_uncert_errors, dtype=float).mean()
mc_der_uncertainty_std = np.array(mc_der_uncert_errors, dtype=float).std()
mc_e_uncertainty = np.array(mc_der_uncert_e_errors, dtype=float).mean()
mc_e_uncertainty_std = np.array(mc_der_uncert_e_errors, dtype=float).std()
mc_a_uncertainty = np.array(mc_der_uncert_a_errors, dtype=float).mean()
mc_a_uncertainty_std = np.array(mc_der_uncert_a_errors, dtype=float).std()
with open('results/' + args.model + '/' + args.task + '/' + ood + '/' + 'test_metrics.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(('mae', 'uncertainty', 'epi_uncertainty', 'ale_uncertainty', 'mc_mae', 'mc_uncertainty',
                     'mc_der_uncertainty', 'mc_e_uncertainty', 'mc_a_uncertainty'))
    for mae_err, uncert_err, epi_uncert_err, ale_uncert_err, mc_mae_err, mc_uncert_err, mc_der_uncert_err, mc_der_uncert_e_err, mc_der_uncert_a_err in zip(
            mae_errors, uncert_errors, epi_uncert_errors,
            ale_uncert_errors, mc_mae_errors, mc_uncert_errors, mc_der_uncert_errors, mc_der_uncert_e_errors,
            mc_der_uncert_a_errors):
        writer.writerow((mae_err, uncert_err, epi_uncert_err, ale_uncert_err, mc_mae_err, mc_uncert_err,
                         mc_der_uncert_err, mc_der_uncert_e_err, mc_der_uncert_a_err))
    writer.writerow(('mae(std)', 'uncertainty(std)', 'epi(std)', 'ale(std)', 'mc_mae(std)', 'mc_uncertainty(std)',
                     'mc_der_uncertainty(std)', 'mc_e(std)', 'mc_a(std)'))
    writer.writerow((f'{mae}({mae_std})', f'{uncertainty}({uncertainty_std})',
                     f'{epi_uncertainty}({epi_uncertainty_std})', f'{ale_uncertainty}({ale_uncertainty_std})',
                     f'{mc_mae}({mc_mae_std})', f'{mc_uncertainty}({mc_uncertainty_std})',
                     f'{mc_der_uncertainty}({mc_der_uncertainty_std})',
                     f'{mc_e_uncertainty}({mc_e_uncertainty_std})', f'{mc_a_uncertainty}({mc_a_uncertainty_std})'))