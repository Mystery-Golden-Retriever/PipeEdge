""" Basic operations used for Quantization """
from typing import List
import numpy as np
import torch

def _quant_op(input_data, bit, e_bit,mode='original'):
    e=int(e_bit)
    m=bit-e-1
    sign=np.sign(input_data).flatten()
    sign=np.where(sign>=0,0,1)
    abs=np.abs(input_data).flatten()
    expmax=int(np.floor(np.log2(np.max(abs))))
    expbias=expmax-(2**e-1)
    valueMin=2**expbias
    valueMax=2**expmax*(2-2**(-1*m))

    # clip orginal data to [valueMin, valueMax]
    abs=np.clip(abs,valueMin,valueMax)
    exp=np.log2(abs)
    exp=np.floor(exp)
    mant=np.floor((abs/(2.0**exp)-1)/(2**-m))
    # compress orginal exponent result to bit-compress-friendly result, using expbias
    exp-=expbias
    int_map=np.vstack((sign,exp,mant)).T.astype(np.uint32)
    return e,expbias,int_map


def _intmap_encode(int_map, bitwidth,e_bit):
    """ compress the converted int_map to tesnor with fewer numbers"""
    # the int_map is assumed as a 4- or 3-dimensional np.array [b(optional),c,h,w]
    int_map = int_map.flatten()
    # enc_ratio is the number of original values compressed into one single int32 value
    enc_ratio = int(32/bitwidth)*3

    # store tensor into new_tensor
    int_map_ext = np.append(int_map,
                            np.repeat(0, (enc_ratio - len(int_map) % enc_ratio) % enc_ratio))
    int_map_rs = np.reshape(int_map_ext, (-1, enc_ratio))
    e=int(e_bit)
    m=bitwidth-e-1
    
    bitshift = np.tile(np.array([0, 1, e + 1]), enc_ratio // 3) + np.repeat(np.arange(0, enc_ratio//3), 3) * bitwidth
    int_map_shifted = np.left_shift(int_map_rs, bitshift)
    new_array = np.bitwise_or.reduce(int_map_shifted, axis=1, dtype=np.uint32)

    return new_array

def _intmap_decode(input_data, orig_shape, e_bit, expbias, bitwidth):
    """ restore the compressed tensor """
    # the input is assumed as an 1-dimensional tensor / np.array

    enc_ratio = int(32/bitwidth)*3
    data_exploded = np.repeat(input_data, enc_ratio)
    data_rs = np.reshape(data_exploded, (-1, enc_ratio))
    
    e=int(e_bit)
    m=bitwidth-e-1
    
    bitshift = np.tile(np.array([0, 1, e + 1]), enc_ratio // 3) + np.repeat(np.arange(0, enc_ratio//3), 3) * bitwidth
    
    data_shifted = np.right_shift(data_rs, bitshift)
    data_flat = data_shifted.flatten()
    
    signals=np.bitwise_and(data_flat[0::3],1)*(-2)+1
    exps=np.bitwise_and(data_flat[1::3],2**e-1)
    mants=np.bitwise_and(data_flat[2::3],2**m-1)
    
    # restore data. For mantissa part, we use central-Round rule.
    orgs=signals*(2.0**(exps+expbias))*((mants+0.5)*(2**-m)+1)
    zerosIndex=(exps==0) & (mants==0)
    orgs[zerosIndex]=0.0
    orgs=orgs[:np.prod(orig_shape)]
    orgs=np.array(orgs).reshape(orig_shape)
    return orgs

# test case for 4 bit 
# expected result should be [[-1.25    2.5    -1.75    0.4375],[-1.25    2.5     1.25    0.    ],[ 0.      0.     -0.875  -0.875 ],[ 0.     -0.4375  0.625  -2.5   ]]

# test=np.array([[-1.17,2.71,-1.6,0.43],[-1.14,2.05,1.01,0.07],[0.16,-0.03,-0.89,-0.87],[-0.04,-0.39,0.64,-2.89]])
# e_bit,expbias,int_map=_quant_op(test,4)
# encoded=_intmap_encode(int_map,4)
# print(_intmap_decode(encoded,test.shape,e_bit,expbias,4))

def _intmap2float(int_map, bitwidth):
    """ used to restore the tesnor from intmap to float """
    scale = (1 << bitwidth) - 1
    return (int_map/scale).astype(np.float32)

def _uint32_to_uint8(tensor):
    """ re-represent uint32 to uint8, since torch has no uint32 (does have uint8) """
    assert tensor.dtype == np.uint32
    return tensor.view('uint8')

def _uint8_to_uint32(tensor):
    """ restore the uint32 value from 4 uint8 values """
    assert tensor.dtype == np.uint8
    return tensor.view('uint32')


def compression_factor(quant_bit: torch.Tensor) -> torch.Tensor:
    """Compute the compression factor (data size improvement) for quantization bit widths > 0."""
    return torch.div(32, quant_bit)


def tensor_encode(input_data: torch.Tensor, quant_bit: int,e_bit:int) -> List[torch.Tensor]:
    """
        The input to the encoder should be a torch.Tensor
        We first cast it to a np.array, then do everything else
    """
    quant_bit_tensor = torch.tensor(quant_bit, dtype = torch.int8)
    if quant_bit == 0:
        return [input_data, torch.tensor(input_data.shape), torch.tensor(0.0),
                quant_bit_tensor]

    input_data = input_data.numpy()
    shape = input_data.shape
    
    # quant
    e_bit, expbias, int_map = _quant_op(input_data, quant_bit,e_bit)
    assert 0<=e_bit<=quant_bit-1
    comm_tensor = _intmap_encode(int_map, quant_bit,e_bit)
    # split uint32 into 4 uint8
    comm_tensor = _uint32_to_uint8(comm_tensor)
    # convert array to tensor for p2p communication
    comm_tensor = torch.tensor(comm_tensor, dtype = torch.uint8)
    shape = torch.tensor(shape, dtype = torch.int32)
    expbias = torch.tensor(expbias, dtype = torch.float32)
    e_bit=torch.tensor(e_bit,dtype=torch.float32)

    return [comm_tensor, shape, e_bit, expbias, quant_bit_tensor]


def tensor_decode(encodings: List[torch.Tensor]) -> torch.Tensor:
    """
        decode the compressed tensor with uint8 value
    """
    comm_tensor, input_shape, e_bit, expbias, quant_bit = encodings
    if quant_bit == 0:
        return comm_tensor

    # convert tensor to array for computation and splice uint8 to uint32
    assert isinstance(comm_tensor, torch.Tensor)
    comm_tensor = _uint8_to_uint32(comm_tensor.numpy())
    input_shape = input_shape.tolist()
    expbias = expbias.item()
    quant_bit = quant_bit.item()
    e_bit=e_bit.item()
    orig_tensor = _intmap_decode(comm_tensor, input_shape, e_bit, expbias, quant_bit)
    return torch.from_numpy(orig_tensor.astype(np.float32))


def tensor_encode_outerdim(batched_tensor: torch.Tensor, quant_bit: int,e_bit:int) -> List[torch.Tensor]:
    """do quantization on each image in the micro-batched tensor with size [b,c,h,w]"""
    list_of_lists = [tensor_encode(t, quant_bit,e_bit) for t in batched_tensor]
    encoded_tensors = list(zip(*list_of_lists))
    return [torch.stack(t,0) for t in encoded_tensors]


def tensor_decode_outerdim(batched_encodings: List[torch.Tensor]) -> torch.Tensor:
    """decode the encoded tensor with multiple images in one batch, each encoded image data is in length of 5"""
    tensors = [tensor_decode(encodings) for encodings in zip(*batched_encodings)]
    return torch.stack(tensors, 0)
