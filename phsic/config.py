def convert_config_for_phsic(kernel_config, encoder_config):
    kernel_config_new = kernel_config[:]
    encoder_config_new = encoder_config[:]

    kernel_type = kernel_config[0]
    encoder_type = encoder_config[0]
    if kernel_type == 'Cos' and encoder_type in {'SumBov', 'AveBov',
                                                 'NormalizedSumBov'}:
        kernel_config_new[0] = 'Linear'
        encoder_config_new[0] = 'NormalizedSumBov'
    elif kernel_type == 'Cos' and encoder_type in {'USE', 'NormalizedUSE'}:
        kernel_config_new[0] = 'Linear'
        encoder_config_new[0] = 'NormalizedUSE'

    return kernel_config_new, encoder_config_new
