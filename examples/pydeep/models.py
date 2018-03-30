model = {
    'inception_v3': {
        'path':  'inception_v3/classify_image_graph_def.pb',
        'bottleneck_tensor_name': 'pool_3/_reshape:0',
        'bottleneck_tensor_size': 2048,
        'jpg_input' : 'jpg_input_data',
        'jpg_input_tensor_name' : 'jpg_input_data:0', 
        'mul_image' : 'mul_image',
        'mul_image_tensor_name' : 'mul_image:0',   
        'input_width': 299,
        'input_height': 299,
        'input_depth': 3,
        'resized_input_tensor_name': 'Mul:0',
        'input_mean': 128,
        'input_std': 128,
        }
    }
