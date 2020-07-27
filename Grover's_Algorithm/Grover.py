from qiskit import *
from qiskit.quantum_info import Operator
from matplotlib import pyplot as plt
from matplotlib import image as img
import ast
from qiskit.visualization import *
import numpy as np


def phase_oracle(N, marked_element):
    qc = QuantumCircuit(N, name="Phase Oracle")
    po = np.eye(2 ** N)
    for x in marked_element:
        po[x, x] = -1
    qc.unitary(Operator(po), range(N))
    return qc


def diffusion_operator(N):
    qc = QuantumCircuit(N, name="Diffusion Operator")
    qc.h(range(N))
    qc.append(phase_oracle(N, [0]), range(N))
    qc.h(range(N))
    return qc


def run_circuit_on_sim(qc):
    print('Running on the simulator...')
    sim = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend=sim, shots=1000000).result()
    print('Running on the simulator finished.')
    return result.get_counts(qc)


def sanpshot(qc, N):
    print('Taking Snapshot... ')
    qc_new = QuantumCircuit(N, N)
    qc_new.append(qc, range(N))
    qc_new.measure(range(N), range(N))
    count = run_circuit_on_sim(qc_new)
    return str(count) + '\n'


def Grover(N, list_of_marked_elements):
    print('Entering Grover\'s algorithm(Quantum Part) \n')
    qc = QuantumCircuit(N, N);

    grovers_iteration = int(np.floor(np.pi / 4 * np.sqrt(2 ** N / len(list_of_marked_elements))))
    print('Number of Grover\'s iteration is (same as number of snapshots) {}'.format(grovers_iteration))
    qc.h(range(N))
    count_string = ''
    count_string += sanpshot(qc, N)
    print(
        'Applying the phase oracle and diffusion operator and taking snapshots after every addition of Grover\'s operator....')
    for _ in range(grovers_iteration):
        qc.append(phase_oracle(N, list_of_marked_elements), range(N))
        qc.append(diffusion_operator(N), range(N))
        count_string += sanpshot(qc, N)

    print('Writing the snapshots to the files...')
    f = open('Grovers.txt', 'w')
    f.write(count_string)
    f.close()
    print('Writing finished')
    qc.measure(range(N), range(N))
    count = run_circuit_on_sim(qc)
    print('Final count for drawing histogram: ' + str(count))

    return grovers_iteration, count


def DrawImage(pmf, ax, row_index):
    print('Drawing each snapshot in memory.')
    column_counter = 0
    for x, y in pmf.items():
        image = img.imread('alpha/{}.png'.format(x))
        ax[row_index][column_counter].imshow(image, alpha=y)
        ax[row_index][column_counter].axis(False)
        column_counter += 1


def visualization(N):
    print('Visulization startred....')
    fp = open('Grovers.txt', 'r')
    raw_snaps = fp.read().split('\n')[:N + 1]
    fp.close()
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    print("Number of snapshots captured from the quantum Cirucit:{}".format(len(raw_snaps)))
    row_counter = 0
    fig, ax = plt.subplots(N + 1, len(alpha))
    for snap in raw_snaps:
        snap_dict = ast.literal_eval(snap)
        new_snap_dict = {int(x, 2): y for x, y in snap_dict.items()}
        new_snap_dict = {chr(x): y for x, y in new_snap_dict.items() if chr(x) in alpha}
        max_prob = max(new_snap_dict.values())
        new_snap_dict = {x: y / max_prob for x, y in
                         sorted(new_snap_dict.items(), key=lambda item: item[1], reverse=True)}
        DrawImage(new_snap_dict, ax, row_counter)
        row_counter += 1
    fig.patch.set_facecolor('#ffff9c')


if __name__ == '__main__':
    search_string = 'S'
    grov_iter, count = Grover(N=8, list_of_marked_elements=[ord(x) for x in search_string.upper()]) # N= number of qubits to be used..
    print('The number of grover\'s iteration  being passed for visualization: {}'.format(grov_iter))
    # grov_iter = 2 # This is to be pasted from the earlier result
    visualization(grov_iter)
    plot_histogram(count)
    # plot_histogram(`count dictionary to be pasted from eaarlier result`)
    plt.show()
