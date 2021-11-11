import numpy as np

conv0_in0_out0 = np.array([
	[-1.0, -1.0, -0.999267578125, 0.23388671875, -1.0],
	[-0.999755859375, 0.67529296875, -1.0, -1.0, -0.000244140625],
	[0.259033203125, -1.0, -1.0, 0.16650390625, -0.0078125],
	[-0.999755859375, 0.156005859375, -0.999755859375, -0.000244140625, -0.0625],
	[-0.039794921875, -0.02587890625, -1.0, -0.019775390625, -1.0]])

conv0_in0_out1 = np.array([
	[0.0, -1.0, -1.0, 0.22509765625, -0.2880859375],
	[-0.00537109375, -0.000244140625, 0.083984375, -1.0, -1.0],
	[-1.0, 0.052734375, -0.03369140625, -1.0, -0.193603515625],
	[0.1337890625, -1.0, -0.000244140625, 0.1240234375, -1.0],
	[-1.0, -1.0, -1.0, -0.96337890625, -0.147705078125]])

conv0_in0_out2 = np.array([
	[0.25146484375, 0.47314453125, 0.1923828125, -0.026123046875, -0.943603515625],
	[0.241943359375, -0.970947265625, -0.9423828125, -0.815673828125, -0.24853515625],
	[0.063232421875, 0.12353515625, 0.192138671875, -0.99853515625, -0.197265625],
	[0.04833984375, -0.015869140625, -0.999267578125, -0.0380859375, -0.998291015625],
	[-0.99609375, -0.998046875, -0.295166015625, -0.285400390625, -0.995849609375]])

conv0_in0_out3 = np.array([
	[0.30859375, 0.01904296875, -0.000732421875, -0.2119140625, -0.544189453125],
	[-1.0, 0.185302734375, -0.99951171875, -0.013916015625, -1.0],
	[-1.0, -1.0, 0.07861328125, -0.052734375, -0.377197265625],
	[0.13330078125, -1.0, -0.000244140625, -1.0, -0.999755859375],
	[-0.011962890625, -1.0, -0.0615234375, -0.100830078125, -0.478515625]])

conv0_in0_out4 = np.array([
	[-1.0, -0.01708984375, -0.015625, -1.0, -1.0],
	[0.198486328125, -0.999267578125, -0.01416015625, -0.00048828125, -1.0],
	[-1.0, -1.0, 0.206787109375, 0.151123046875, -1.0],
	[-1.0, -1.0, -0.99951171875, -1.0, -1.0],
	[-0.000244140625, -1.0, -0.999755859375, 0.137451171875, -1.0]])

conv0_in0_out5 = np.array([
	[-0.997314453125, -0.999267578125, 0.4267578125, -0.999755859375, -0.00048828125],
	[-0.99951171875, -0.999755859375, -1.0, -0.999755859375, -0.99951171875],
	[0.093017578125, -1.0, -0.998779296875, -1.0, -0.00048828125],
	[-0.008056640625, 0.11767578125, 0.0478515625, -0.000732421875, -1.0],
	[-0.226318359375, -0.0838623046875, -0.045166015625, -0.0048828125, -0.999755859375]])

weights_conv0_out0 = np.array([
	conv0_in0_out0])

weights_conv0_out1 = np.array([
	conv0_in0_out1])

weights_conv0_out2 = np.array([
	conv0_in0_out2])

weights_conv0_out3 = np.array([
	conv0_in0_out3])

weights_conv0_out4 = np.array([
	conv0_in0_out4])

weights_conv0_out5 = np.array([
	conv0_in0_out5])

weights_conv0 = np.array([
	weights_conv0_out0,
	weights_conv0_out1,
	weights_conv0_out2,
	weights_conv0_out3,
	weights_conv0_out4,
	weights_conv0_out5])

conv1_in0_out0 = np.array([
	[-1.0, -0.914306640625, -0.9306640625, 0.738525390625, 0.7568359375],
	[-1.0, 0.686279296875, 0.675537109375, -0.999755859375, 0.75634765625],
	[0.658935546875, 0.7578125, -0.998291015625, -0.998779296875, -1.0],
	[0.657958984375, -1.0, -0.994873046875, -0.999755859375, 0.718017578125],
	[-0.87353515625, -0.982421875, 0.916259765625, -1.0, 0.64892578125]])

conv1_in0_out1 = np.array([
	[-0.9990234375, 0.759765625, -1.0, -1.0, -0.99951171875],
	[-0.99951171875, -1.0, -0.999755859375, 0.7958984375, -1.0],
	[-1.0, -1.0, 0.52392578125, -1.0, 0.75732421875],
	[0.71630859375, 0.752197265625, -0.999755859375, 0.57470703125, -1.0],
	[-1.0, -1.0, 0.7119140625, -1.0, -1.0]])

conv1_in0_out2 = np.array([
	[-0.998779296875, -0.99267578125, 0.970458984375, -0.9755859375, 0.917236328125],
	[-1.0, -1.0, -1.0, -0.99951171875, 0.75390625],
	[-0.99951171875, -1.0, 0.845458984375, 0.817138671875, 0.74462890625],
	[-1.0, 0.761474609375, -1.0, 0.767333984375, 0.76513671875],
	[-1.0, 0.796875, -0.999755859375, -0.99951171875, 0.82861328125]])

conv1_in0_out3 = np.array([
	[0.312255859375, -0.999755859375, -0.999755859375, -0.999755859375, -1.0],
	[-1.0, 0.917236328125, 0.91357421875, 0.818115234375, -1.0],
	[-0.999755859375, -0.99951171875, -1.0, 0.5576171875, -1.0],
	[-0.998291015625, 0.590087890625, 0.472412109375, -1.0, 0.59423828125],
	[0.641357421875, -0.999755859375, -1.0, 0.676513671875, 0.81787109375]])

conv1_in0_out4 = np.array([
	[-1.0, 0.834228515625, -1.0, 0.816162109375, 0.88525390625],
	[-0.999755859375, 0.85791015625, 0.762451171875, -0.999755859375, -0.99609375],
	[-1.0, 0.943603515625, -1.0, 0.83740234375, 0.839599609375],
	[-1.0, 0.94287109375, 0.91845703125, -0.999755859375, 0.83154296875],
	[0.8818359375, 0.918212890625, -1.0, -0.999755859375, 0.78173828125]])

conv1_in0_out5 = np.array([
	[0.960205078125, -1.0, -1.0, -0.997314453125, 0.704833984375],
	[0.994140625, 0.943359375, 0.880615234375, -0.999755859375, 0.676025390625],
	[-1.0, 0.981201171875, 0.83251953125, 0.6904296875, -0.999755859375],
	[0.986572265625, 0.9345703125, 0.769287109375, -1.0, 0.6630859375],
	[0.9814453125, -0.997802734375, 0.734375, -0.99853515625, -1.0]])

conv1_in0_out6 = np.array([
	[0.242431640625, 0.4345703125, 0.5400390625, 0.498291015625, -1.0],
	[0.435302734375, 0.623779296875, 0.579345703125, 0.4287109375, 0.632080078125],
	[0.607177734375, -1.0, -0.999755859375, -1.0, 0.69287109375],
	[-1.0, 0.512451171875, -1.0, -0.99951171875, -1.0],
	[0.40673828125, 0.388916015625, -1.0, 0.533935546875, 0.57861328125]])

conv1_in0_out7 = np.array([
	[0.779296875, -1.0, 0.31884765625, -1.0, 0.331787109375],
	[0.73095703125, 0.6484375, 0.328857421875, 0.273681640625, -0.999755859375],
	[-0.999755859375, -0.999755859375, -0.9990234375, -1.0, -1.0],
	[-0.999755859375, -1.0, 0.714111328125, -1.0, 0.316650390625],
	[0.7109375, 0.6162109375, -1.0, -0.99951171875, 0.216796875]])

conv1_in1_out0 = np.array([
	[-1.0, -1.0, 0.170654296875, -1.0, -1.0],
	[0.02490234375, -0.00048828125, -1.0, -0.999755859375, -1.0],
	[-1.0, -0.0029296875, 0.08544921875, -0.999267578125, -1.0],
	[-1.0, -0.005126953125, -1.0, -0.00244140625, -0.017578125],
	[-0.99951171875, -1.0, -0.002197265625, -1.0, -1.0]])

conv1_in1_out1 = np.array([
	[-0.999755859375, -0.999755859375, -0.998779296875, -0.999755859375, -0.9990234375],
	[0.050048828125, -1.0, -0.00537109375, -0.999755859375, -1.0],
	[0.11279296875, -0.999755859375, -0.04052734375, -0.00048828125, -1.0],
	[0.053466796875, -1.0, -0.0009765625, -0.001953125, -0.012939453125],
	[0.00732421875, 0.045654296875, -0.00634765625, -1.0, -0.045654296875]])

conv1_in1_out2 = np.array([
	[-1.0, -0.99951171875, 0.611572265625, -1.0, -1.0],
	[-1.0, 0.44580078125, -1.0, -1.0, -0.006103515625],
	[-0.9990234375, 0.12841796875, -0.003662109375, -1.0, -0.99951171875],
	[0.41943359375, -0.009033203125, -0.999755859375, -1.0, -0.010498046875],
	[-1.0, -1.0, -0.000244140625, -1.0, -1.0]])

conv1_in1_out3 = np.array([
	[-0.044189453125, -0.08935546875, -1.0, -0.153076171875, -0.149169921875],
	[-1.0, -1.0, -0.000732421875, -1.0, -1.0],
	[0.261962890625, 0.088134765625, 0.0576171875, -1.0, -0.014892578125],
	[0.352783203125, -1.0, 0.077392578125, -1.0, -1.0],
	[-0.9990234375, 0.037353515625, -0.003662109375, -1.0, -0.08935546875]])

conv1_in1_out4 = np.array([
	[-1.0, -0.999267578125, -1.0, -0.00048828125, -1.0],
	[-1.0, -1.0, -1.0, -0.021728515625, -0.080810546875],
	[-0.000244140625, -1.0, 0.013427734375, -0.001953125, -0.073486328125],
	[-0.99853515625, -1.0, 0.28515625, -1.0, -0.99951171875],
	[-1.0, -0.999267578125, -1.0, -1.0, -1.0]])

conv1_in1_out5 = np.array([
	[-0.997802734375, -1.0, -0.00048828125, -0.002685546875, -0.998779296875],
	[0.186279296875, -0.000244140625, -0.99853515625, -0.99951171875, -0.99951171875],
	[0.159423828125, -0.002197265625, -0.997314453125, -0.998779296875, -1.0],
	[0.031005859375, 0.071533203125, -0.999755859375, -0.029541015625, -0.064208984375],
	[-0.999755859375, 0.126220703125, 0.023681640625, -1.0, -0.999755859375]])

conv1_in1_out6 = np.array([
	[-1.0, -1.0, -0.99951171875, -0.999755859375, -0.012939453125],
	[0.083984375, -0.013427734375, -1.0, -0.999755859375, -0.00146484375],
	[-0.999755859375, -0.013427734375, 0.035888671875, -0.00244140625, -0.022705078125],
	[-0.00048828125, -0.999755859375, -1.0, -0.02099609375, -1.0],
	[-1.0, -0.999755859375, -1.0, -0.049072265625, -1.0]])

conv1_in1_out7 = np.array([
	[-1.0, -1.0, -0.99951171875, -0.01318359375, -0.010009765625],
	[0.396240234375, 0.522216796875, -1.0, -0.02294921875, -0.0517578125],
	[-0.999755859375, -1.0, -1.0, -0.99951171875, -0.02587890625],
	[0.08544921875, -0.999755859375, 0.31982421875, -0.004150390625, -0.998779296875],
	[-0.005615234375, -0.99951171875, 0.244384765625, -0.003173828125, -1.0]])

conv1_in2_out0 = np.array([
	[-0.000244140625, -1.0, -1.0, -0.016845703125, -1.0],
	[-1.0, -0.999755859375, -0.0888671875, -0.02880859375, -1.0],
	[-0.001708984375, -1.0, -0.999755859375, -0.99951171875, -1.0],
	[-0.00341796875, -0.032470703125, -1.0, -0.008544921875, -0.9990234375],
	[-0.999755859375, -1.0, -0.02783203125, -1.0, -1.0]])

conv1_in2_out1 = np.array([
	[-0.00048828125, -0.000244140625, -1.0, -1.0, -0.00927734375],
	[-1.0, -0.006103515625, -0.00146484375, -0.001953125, -0.01513671875],
	[-0.999755859375, -0.000244140625, -0.064697265625, -1.0, -0.999755859375],
	[-0.004638671875, -1.0, -0.999755859375, -1.0, -0.999755859375],
	[-0.00634765625, -1.0, -0.999755859375, -0.999267578125, -0.999755859375]])

conv1_in2_out2 = np.array([
	[-1.0, -1.0, -1.0, -0.000732421875, -0.99951171875],
	[-1.0, -1.0, -0.001953125, -0.999755859375, -0.010498046875],
	[-1.0, -1.0, -0.999267578125, -1.0, -1.0],
	[-1.0, -0.0029296875, -0.050537109375, -0.02783203125, -0.1201171875],
	[-0.999755859375, -1.0, -1.0, -0.02392578125, -0.236083984375]])

conv1_in2_out3 = np.array([
	[-1.0, -1.0, -1.0, -1.0, -0.100341796875],
	[-1.0, -0.999755859375, -1.0, -1.0, -0.998291015625],
	[-0.999755859375, -1.0, -0.00634765625, -0.999755859375, -0.010009765625],
	[-1.0, -1.0, -0.99951171875, -0.026123046875, -0.0234375],
	[-0.0029296875, -0.999755859375, -0.09033203125, -1.0, -0.209716796875]])

conv1_in2_out4 = np.array([
	[-0.000244140625, -0.023193359375, -0.000732421875, -0.0390625, -1.0],
	[-0.010498046875, -0.000244140625, -0.999755859375, -0.00830078125, -0.01611328125],
	[-0.0146484375, -0.00048828125, -1.0, -0.027099609375, -0.999755859375],
	[-1.0, -0.0009765625, -1.0, -1.0, -0.999755859375],
	[-0.012451171875, -0.037353515625, -0.001953125, -0.999755859375, -0.06396484375]])

conv1_in2_out5 = np.array([
	[-0.001220703125, -0.000244140625, -0.00048828125, -0.998779296875, -0.0576171875],
	[-0.99755859375, -0.9990234375, -0.001953125, -0.00244140625, -0.999755859375],
	[-0.006103515625, -1.0, -1.0, -0.001220703125, -1.0],
	[-1.0, -1.0, -0.009521484375, -1.0, -0.116943359375],
	[-1.0, -0.012939453125, -0.999267578125, -0.999267578125, -0.12548828125]])

conv1_in2_out6 = np.array([
	[-0.137451171875, -0.01416015625, -0.003173828125, -0.002685546875, -1.0],
	[-0.999755859375, -1.0, -0.999267578125, -1.0, -1.0],
	[-1.0, -0.999755859375, -1.0, -0.999755859375, -1.0],
	[-0.015625, -0.99951171875, -1.0, -0.9990234375, -0.010986328125],
	[-0.002685546875, -0.007080078125, -0.012451171875, -0.038330078125, -0.05712890625]])

conv1_in2_out7 = np.array([
	[-0.999755859375, -0.999755859375, -1.0, -0.012451171875, -1.0],
	[-1.0, -1.0, -0.032470703125, -1.0, -0.0458984375],
	[-0.000244140625, -1.0, -0.000732421875, -0.01416015625, -0.01806640625],
	[-1.0, -1.0, -1.0, -0.9990234375, -1.0],
	[-0.04931640625, -0.002197265625, -0.999755859375, -0.999755859375, -0.999755859375]])

conv1_in3_out0 = np.array([
	[-0.999755859375, -0.000732421875, -1.0, 0.025146484375, -1.0],
	[-0.02734375, -1.0, -0.01904296875, 0.045166015625, 0.189453125],
	[-0.011474609375, -0.05126953125, -0.999755859375, -0.999755859375, 0.05224609375],
	[-0.999755859375, -0.03125, -0.999755859375, -1.0, -0.00537109375],
	[0.02734375, 0.068603515625, -0.000244140625, 0.041748046875, -1.0]])

conv1_in3_out1 = np.array([
	[-1.0, -0.044921875, -0.000244140625, -1.0, 0.11962890625],
	[-1.0, -0.028076171875, -0.999755859375, -0.99951171875, -1.0],
	[-1.0, -0.043212890625, -0.12890625, -0.005859375, -1.0],
	[-0.999755859375, -1.0, -0.99951171875, -0.025146484375, 0.083251953125],
	[-1.0, -1.0, -0.009765625, -1.0, -0.999755859375]])

conv1_in3_out2 = np.array([
	[-1.0, 0.835205078125, -1.0, -1.0, -1.0],
	[-1.0, 0.50927734375, -0.999755859375, -1.0, -1.0],
	[-0.999267578125, 0.12841796875, -0.042724609375, -0.00830078125, -1.0],
	[0.513427734375, -0.0263671875, -0.103515625, -0.09521484375, -0.09765625],
	[-0.99951171875, -1.0, -1.0, -0.0966796875, -0.079833984375]])

conv1_in3_out3 = np.array([
	[-0.05810546875, -0.01220703125, -0.999755859375, -0.11328125, -0.19677734375],
	[-0.99951171875, -0.999755859375, -1.0, -0.00146484375, -0.036865234375],
	[-0.999755859375, -0.999755859375, -1.0, -0.0322265625, -0.04736328125],
	[-0.997802734375, -0.999755859375, -0.0166015625, -0.045166015625, -0.06689453125],
	[0.237548828125, -0.002685546875, -1.0, -0.02783203125, -0.091796875]])

conv1_in3_out4 = np.array([
	[-1.0, -0.063232421875, -1.0, -1.0, -0.022216796875],
	[-0.999755859375, -0.999267578125, -0.999755859375, -1.0, -1.0],
	[-0.99951171875, -1.0, -0.00146484375, -0.999755859375, -0.118896484375],
	[-0.999755859375, -0.999755859375, 0.131103515625, -1.0, -1.0],
	[-0.005126953125, -0.025146484375, -0.99951171875, -1.0, -0.0126953125]])

conv1_in3_out5 = np.array([
	[-0.05029296875, -0.073974609375, -0.12890625, -0.160888671875, -0.120849609375],
	[-0.001220703125, -0.056396484375, -0.115234375, -0.141357421875, -0.999755859375],
	[-1.0, -0.03564453125, -0.99951171875, -0.092529296875, -0.18310546875],
	[-0.05615234375, -0.999755859375, -0.072998046875, -0.16162109375, -0.189697265625],
	[-0.9990234375, -0.073486328125, -0.122802734375, -0.998291015625, -1.0]])

conv1_in3_out6 = np.array([
	[-0.999267578125, -0.001220703125, -1.0, -0.044189453125, -0.000732421875],
	[-0.0009765625, -1.0, 0.004150390625, -0.002685546875, -0.99951171875],
	[-0.038330078125, -1.0, -1.0, -0.998779296875, -0.0341796875],
	[-0.080810546875, -0.071044921875, -0.033203125, -0.1083984375, -0.050537109375],
	[-1.0, -0.99951171875, -0.999755859375, -1.0, -1.0]])

conv1_in3_out7 = np.array([
	[0.27587890625, -1.0, -1.0, -0.02197265625, -1.0],
	[-1.0, -1.0, 0.0927734375, -0.101318359375, -1.0],
	[-0.999755859375, -1.0, -1.0, -0.025634765625, -0.031982421875],
	[-0.017578125, 0.1044921875, -0.999755859375, -0.01806640625, -0.99951171875],
	[-0.080810546875, -0.011474609375, -1.0, -1.0, -0.013427734375]])

conv1_in4_out0 = np.array([
	[-1.0, -1.0, 0.65771484375, -1.0, 0.817138671875],
	[0.932373046875, 0.8388671875, 0.736328125, -1.0, -1.0],
	[-0.9990234375, 0.740966796875, 0.80615234375, 0.882080078125, 0.885009765625],
	[-0.999755859375, 0.705810546875, -0.999267578125, 0.80224609375, -1.0],
	[-0.999755859375, -0.999755859375, -1.0, -0.99951171875, -0.99267578125]])

conv1_in4_out1 = np.array([
	[0.861328125, 0.81494140625, 0.744873046875, -0.999755859375, 0.786376953125],
	[-1.0, -1.0, 0.762939453125, -0.999755859375, -1.0],
	[-0.999755859375, 0.805908203125, 0.74365234375, 0.64208984375, 0.765380859375],
	[-0.9990234375, -1.0, 0.83056640625, 0.662841796875, 0.770263671875],
	[0.832275390625, -1.0, 0.802734375, 0.781982421875, -1.0]])

conv1_in4_out2 = np.array([
	[-1.0, 0.934326171875, -0.999267578125, 0.953369140625, 0.898681640625],
	[0.95068359375, 0.876220703125, -0.999755859375, 0.902099609375, -1.0],
	[-0.999267578125, 0.81103515625, -1.0, -1.0, -1.0],
	[0.930419921875, -0.999267578125, -1.0, -1.0, 0.912109375],
	[0.92578125, 0.8134765625, -0.999755859375, -1.0, 0.895751953125]])

conv1_in4_out3 = np.array([
	[0.860107421875, 0.78515625, -1.0, -1.0, -1.0],
	[-1.0, 0.7861328125, -1.0, 0.83642578125, 0.83984375],
	[0.90283203125, 0.83740234375, -1.0, -1.0, -0.97802734375],
	[0.822265625, -0.999755859375, 0.800048828125, 0.86767578125, 0.864501953125],
	[-0.99951171875, -0.999755859375, -0.999755859375, -0.999755859375, -1.0]])

conv1_in4_out4 = np.array([
	[-1.0, 0.64453125, -1.0, -0.999755859375, -1.0],
	[0.769775390625, 0.719482421875, -0.999267578125, 0.506103515625, -1.0],
	[0.702392578125, -1.0, -0.99951171875, 0.6962890625, 0.7265625],
	[0.669189453125, -1.0, -0.999267578125, -0.999267578125, -1.0],
	[0.686767578125, -1.0, 0.762451171875, -1.0, 0.81640625]])

conv1_in4_out5 = np.array([
	[0.931396484375, 0.885498046875, 0.83056640625, -1.0, 0.8984375],
	[-1.0, -1.0, 0.890625, -0.99951171875, -0.99951171875],
	[0.9853515625, 0.98095703125, 0.95263671875, 0.893798828125, 0.85791015625],
	[-0.998779296875, 0.997802734375, 0.938720703125, -0.99951171875, -1.0],
	[1.0, -0.998046875, 0.91455078125, 0.802490234375, -1.0]])

conv1_in4_out6 = np.array([
	[0.85595703125, -0.999755859375, 0.787841796875, -0.999755859375, -1.0],
	[-1.0, -0.999267578125, 0.91259765625, 0.79638671875, -1.0],
	[0.809326171875, 0.830322265625, 0.8369140625, -1.0, -1.0],
	[0.830322265625, -1.0, -1.0, 0.817626953125, -1.0],
	[0.890869140625, -0.98486328125, -1.0, -1.0, -0.999755859375]])

conv1_in4_out7 = np.array([
	[-1.0, -0.9990234375, 0.894775390625, -0.999755859375, 0.91552734375],
	[0.770751953125, 0.947265625, -0.99951171875, 0.690185546875, -1.0],
	[0.812255859375, -0.99951171875, -0.9990234375, 0.731201171875, -0.999755859375],
	[-0.999267578125, -0.999755859375, 0.718994140625, 0.79541015625, 0.823974609375],
	[0.841552734375, 0.733154296875, 0.74853515625, 0.828125, 0.831298828125]])

conv1_in5_out0 = np.array([
	[0.43505859375, 0.4677734375, 0.541015625, -0.999755859375, 0.725830078125],
	[-1.0, 0.447998046875, -1.0, 0.599853515625, -1.0],
	[-1.0, -1.0, -0.999755859375, 0.811279296875, -0.999755859375],
	[0.492431640625, 0.644287109375, -1.0, -0.999755859375, -0.999267578125],
	[0.67236328125, -0.999755859375, -0.9716796875, -1.0, 0.727783203125]])

conv1_in5_out1 = np.array([
	[-0.999755859375, -0.99951171875, 0.5556640625, -0.99951171875, -1.0],
	[-0.999755859375, 0.54248046875, -1.0, -1.0, 0.671142578125],
	[0.5458984375, 0.408203125, 0.383544921875, 0.537353515625, 0.67529296875],
	[-1.0, 0.534423828125, -0.999755859375, 0.5205078125, -1.0],
	[-1.0, -0.999755859375, 0.536865234375, 0.579833984375, 0.722412109375]])

conv1_in5_out2 = np.array([
	[-1.0, -0.981201171875, -0.955810546875, -0.95947265625, -0.979736328125],
	[0.976318359375, -1.0, -0.999755859375, -1.0, 0.80859375],
	[0.943115234375, 0.901123046875, -0.998779296875, -1.0, -1.0],
	[-0.999755859375, -0.9990234375, -1.0, 0.764892578125, 0.725341796875],
	[-0.99951171875, -1.0, 0.782470703125, -0.998779296875, -1.0]])

conv1_in5_out3 = np.array([
	[-0.999755859375, 0.50390625, -1.0, -1.0, 0.755615234375],
	[0.57177734375, -1.0, -1.0, -1.0, 0.810546875],
	[-1.0, -0.99951171875, 0.5693359375, 0.575927734375, 0.668701171875],
	[0.77880859375, 0.517822265625, 0.46484375, 0.54833984375, -1.0],
	[-1.0, 0.60107421875, -0.99951171875, 0.801513671875, 0.847900390625]])

conv1_in5_out4 = np.array([
	[-1.0, 0.806640625, -1.0, 0.850830078125, -1.0],
	[-1.0, 0.833740234375, 0.891845703125, -0.999755859375, -1.0],
	[0.8994140625, -1.0, -1.0, 0.833251953125, 0.896484375],
	[-0.99951171875, 0.916015625, -1.0, -0.999755859375, -1.0],
	[0.8427734375, -1.0, -1.0, 0.889404296875, 0.778564453125]])

conv1_in5_out5 = np.array([
	[-1.0, -0.99853515625, -1.0, 0.884521484375, -0.994384765625],
	[0.9638671875, 0.933837890625, -0.998046875, 0.814453125, 0.720703125],
	[0.987548828125, 0.964599609375, -1.0, 0.777099609375, -0.998291015625],
	[0.9873046875, 0.963134765625, 0.85009765625, 0.7646484375, -1.0],
	[0.98388671875, 0.9443359375, -0.9990234375, -0.999267578125, 0.696044921875]])

conv1_in5_out6 = np.array([
	[-1.0, 0.23193359375, -0.99951171875, 0.462890625, 0.570556640625],
	[0.4443359375, 0.521240234375, 0.406005859375, 0.302001953125, 0.4384765625],
	[0.654541015625, -1.0, -0.998291015625, 0.36962890625, 0.52294921875],
	[-0.999267578125, 0.32177734375, -0.99951171875, -1.0, -1.0],
	[0.314697265625, -0.999755859375, 0.246337890625, -0.998779296875, 0.447021484375]])

conv1_in5_out7 = np.array([
	[-1.0, -0.99951171875, -1.0, -0.999755859375, -0.99951171875],
	[0.724609375, -1.0, 0.38525390625, 0.622802734375, -0.99951171875],
	[-0.998046875, -1.0, 0.4609375, -1.0, -1.0],
	[0.83203125, -0.999267578125, -0.999755859375, -1.0, 0.464599609375],
	[0.742919921875, 0.67724609375, 0.589111328125, 0.400634765625, 0.35693359375]])

weights_conv1_out0 = np.array([
	conv1_in0_out0,
	conv1_in1_out0,
	conv1_in2_out0,
	conv1_in3_out0,
	conv1_in4_out0,
	conv1_in5_out0])

weights_conv1_out1 = np.array([
	conv1_in0_out1,
	conv1_in1_out1,
	conv1_in2_out1,
	conv1_in3_out1,
	conv1_in4_out1,
	conv1_in5_out1])

weights_conv1_out2 = np.array([
	conv1_in0_out2,
	conv1_in1_out2,
	conv1_in2_out2,
	conv1_in3_out2,
	conv1_in4_out2,
	conv1_in5_out2])

weights_conv1_out3 = np.array([
	conv1_in0_out3,
	conv1_in1_out3,
	conv1_in2_out3,
	conv1_in3_out3,
	conv1_in4_out3,
	conv1_in5_out3])

weights_conv1_out4 = np.array([
	conv1_in0_out4,
	conv1_in1_out4,
	conv1_in2_out4,
	conv1_in3_out4,
	conv1_in4_out4,
	conv1_in5_out4])

weights_conv1_out5 = np.array([
	conv1_in0_out5,
	conv1_in1_out5,
	conv1_in2_out5,
	conv1_in3_out5,
	conv1_in4_out5,
	conv1_in5_out5])

weights_conv1_out6 = np.array([
	conv1_in0_out6,
	conv1_in1_out6,
	conv1_in2_out6,
	conv1_in3_out6,
	conv1_in4_out6,
	conv1_in5_out6])

weights_conv1_out7 = np.array([
	conv1_in0_out7,
	conv1_in1_out7,
	conv1_in2_out7,
	conv1_in3_out7,
	conv1_in4_out7,
	conv1_in5_out7])

weights_conv1 = np.array([
	weights_conv1_out0,
	weights_conv1_out1,
	weights_conv1_out2,
	weights_conv1_out3,
	weights_conv1_out4,
	weights_conv1_out5,
	weights_conv1_out6,
	weights_conv1_out7])

