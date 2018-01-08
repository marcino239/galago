import numpy as np
import struct

#from matplotlib import pyplot as plt

N = 2048        # needs to be even
MAX_NUM = 5
MAX_EPOCHS = 1000
MUTATION_PROBABILITY = 0.5  
GENE_BIT_LEN = 64 * 2
LARGE_VAL = 1e20
ELITE_RATIO = 0.1

T = np.arange( 0, 2.0 * np.pi, 0.01 )
Y = np.sin( np.pi * T + np.pi / 128 )

def encode( w, p ):
    b = struct.pack( 'dd', w, p )
    return b
    
def decode( b ):
    w, p = struct.unpack( 'dd', b )
    return w, p

def random_gene():
    return encode( (np.random.random() - 0.5) * MAX_NUM, (np.random.random() - 0.5) * MAX_NUM )

def mutate( b, bits ):
    a = bytearray( b )
    for bit in bits:
        a[ bit >> 3 ] ^= 1 << (bit & 7)
    
    return a

def crossover( b1, b2, cut = None ):
    
    if cut is None:
        cut = np.random.randint( 0, len( b1 ) >> 3 )
    byte = cut >> 3
    bit = cut & 7

    def single_byte( byte1, byte2, bit_pos ):
        return (byte1 & ~((1 << bit_pos)-1)) | (byte2 & (1 << bit_pos) - 1)

    t1 = bytearray( 1 )
    t1[ 0 ] = single_byte( b1[byte], b2[byte], bit )
    if byte <= 0:
        t = t1 + bytearray( b2[ byte+1: ] )
    elif byte + 1 > len( b1 ):
        t = bytearray( b1[ : byte ] ) + t1
    else:
        t = bytearray( b1[ : byte ] ) + t1 + bytearray( b2[ byte+1: ] )

    assert len( t ) == 16

    return t

def fitness( b ):
    w, p = decode( b )
    y = np.sin( T * w + p )
    
    y = list( map( lambda x: LARGE_VAL if np.isnan( x ) else x, y ) )
    
    fit = np.sum( (Y - y) ** 2 )
    return fit
    
def mate( herd, ELITE_RATIO, MUTATION_PROBABILITY ):
    esize = int( N * ELITE_RATIO )
    res = herd[ :esize ]
    
    def gen():
        for i in range( esize, N ):
            b1 = np.random.randint(0, N / 2)
            b2 = np.random.randint(0, N / 2)
            spos = np.random.randint(0, GENE_BIT_LEN )
            
            t = crossover( herd[b1], herd[b2], spos )
            
            if np.random.random() < MUTATION_PROBABILITY:
                t = mutate( t, np.random.randint( GENE_BIT_LEN, size=1 ) )
            
            assert len( t ) == 16
            yield t
            
    res = res + [ e for e in gen() ]
    return res

def run_optimisation( N=N,
            MAX_EPOCHS=MAX_EPOCHS,
            LARGE_VAL=LARGE_VAL,
            ELITE_RATIO=ELITE_RATIO,
            MUTATION_PROBABILITY=MUTATION_PROBABILITY,
            diag=True, seed=None ):

    # set the rand state
    if seed is not None:
        np.random.seed( 0 )

    # algo
    #   1. generate initial population
    herd = [ random_gene() for i in range( N ) ]

    perf = []

    for epoch in range( MAX_EPOCHS ):

        #   5. for each element calculate fitness
        fit = list( map( fitness, herd ) )

        #   6. sort herd by fitness
        keys = sorted( range( N ), key = lambda x: fit[ x ] )
        herd = [ herd[ k ] for k in keys ]

        herd = mate( herd, ELITE_RATIO, MUTATION_PROBABILITY )

        # pull best gene
        best_gene = herd[ 0 ]
        best_gene_fit = fit[ keys[0] ]
        
        perf.append( (epoch, best_gene_fit) )
        
        # diagnostics
        if diag:
            print( '{0}:\t{1}\t{2},{3}'.format( epoch, best_gene_fit, *decode( best_gene ) ) )

    return herd[ 0 ], perf


if __name__ == '__main__':
    run_optimisation( seed = 0 )
