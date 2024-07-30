from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy

class RDKitMoleculeDrawer:
    def __init__(self, method, smiles, start_node_idx, end_node_idx, mid_node_idx, score, prediction, sparsity, fidelity, fidelity_inv):
        self.method = method
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(self.mol)
        self.atom_coords = [self.mol.GetConformer().GetAtomPosition(i) for i in range(self.mol.GetNumAtoms())]
        self.bond_coords = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in self.mol.GetBonds()]
        self.start_node_idx = start_node_idx
        self.end_node_idx = end_node_idx
        self.mid_node_idx = mid_node_idx
        self.score = score
        self.prediction = prediction
        self.sparsity = sparsity
        self.fidelity = fidelity
        self.fidelity_inv = fidelity_inv
        
    def draw(self, ax):
        edge_count = len(self.bond_coords)
       
        for start_idx, end_idx in self.bond_coords:
            start_coords = self.atom_coords[start_idx]
            end_coords = self.atom_coords[end_idx]

            delta_x = end_coords[0] - start_coords[0]
            delta_y = end_coords[1] - start_coords[1]
            start_x = start_coords[0] + delta_x * 0.2
            start_y = start_coords[1] + delta_y * 0.2
            end_x = end_coords[0] - delta_x * 0.2
            end_y = end_coords[1] - delta_y * 0.2

            line = Line2D([start_x, end_x], [start_y, end_y], color='black', linestyle='dashed', linewidth=2)
            ax.add_line(line)

        start_coord = self.atom_coords[self.start_node_idx]
        mid_coord = self.atom_coords[self.mid_node_idx]
        end_coord = self.atom_coords[self.end_node_idx]
       
        start_x, start_y = start_coord.x, start_coord.y
        mid_x, mid_y = mid_coord.x, mid_coord.y
        end_x, end_y = end_coord.x, end_coord.y

        # Çizgiyi çizin
        R = self.score
        alpha = abs(R) * 5
        if alpha > 1.0:
            alpha = 1.0
       
        line_color = 'black'
       
        if R > 0.0:
            line_color = 'red'
        elif R < -0.0:
            line_color = 'blue'
       
       
        line1 = Line2D([start_x - 0.1, mid_x - 0.1], [start_y, mid_y], color=line_color, alpha=alpha)
        line2 = Line2D([mid_x - 0.1, end_x - 0.1], [mid_y, end_y], color=line_color, alpha=alpha)

        ax.add_line(line1)
        ax.add_line(line2)

        ax.scatter(mid_x - 0.1, mid_y, color='white', s=200)
       
       
        for i, atom_coord in enumerate(self.atom_coords):
            atom_symbol = self.mol.GetAtomWithIdx(i).GetSymbol()  # Atom sembolünü al
            if i in [self.start_node_idx, self.end_node_idx, self.mid_node_idx]:
                circle = Circle((atom_coord[0] - 0.1, atom_coord[1]), radius=0.1, color=line_color, alpha=alpha)
                ax.add_patch(circle)
            ax.text(atom_coord[0] - 0.1, atom_coord[1], atom_symbol, fontsize=10, ha='center', va='center')
            
        
        mol_xlim = (min(atom_coord[0] for atom_coord in self.atom_coords) - 0.1, 
            max(atom_coord[0] for atom_coord in self.atom_coords) + 0.1)
        mol_ylim = (min(atom_coord[1] for atom_coord in self.atom_coords) - 0.1, 
                    max(atom_coord[1] for atom_coord in self.atom_coords) + 0.1)
        
        # Ax'in boyutunu çizimin boyutundan daha büyük ayarla
        ax.set_xlim(mol_xlim[0] - 5, mol_xlim[1] + 0.5)  # Örneğin 0.5 birim daha büyük
        ax.set_ylim(mol_ylim[0] - 0.5, mol_ylim[1] + 3.2) 
        
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+3.0, f'Method: {self.method}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+2.5, f'Smiles: {self.smiles}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+2.0, f'Prediction: {self.prediction}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+1.5, f'Sparsity: {self.sparsity:.4f}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+1.0, f'Fidelity: {self.fidelity:.4f}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+0.5, f'Fidelity_inv: {self.fidelity_inv:.4f}', fontsize=10, ha='left', va='center')
        ax.set_axis_off()
        
    def draw_bond_between_atoms(self, ax):
        
        for start_idx, end_idx in self.bond_coords:
            start_coords = self.atom_coords[start_idx]
            end_coords = self.atom_coords[end_idx]

            start_x, start_y = start_coords[0], start_coords[1]
            end_x, end_y = end_coords[0], end_coords[1]

            line = Line2D([start_x, end_x], [start_y, end_y], color='black', linestyle='dashed', linewidth=2)
            ax.add_line(line)
        

        start_coord = self.atom_coords[self.start_node_idx]
        mid_coord = self.atom_coords[self.mid_node_idx]
        end_coord = self.atom_coords[self.end_node_idx]
       
        start_x, start_y = start_coord.x, start_coord.y
        mid_x, mid_y = mid_coord.x, mid_coord.y
        end_x, end_y = end_coord.x, end_coord.y

        # Çizgiyi çizin
        R = self.score
        alpha = abs(R)
        #print("alpha", alpha)
        if alpha > 1.0:
            alpha = 1.0
       
        line_color = 'red'
       
       
        line1 = Line2D([start_x, end_x], [start_y, end_y], color=line_color, alpha=alpha)
        
        ax.add_line(line1)
        
        ax.scatter(mid_x - 0.1, mid_y, color='white', s=200)
       
       
        for i, atom_coord in enumerate(self.atom_coords):
            atom_symbol = self.mol.GetAtomWithIdx(i).GetSymbol()  # Atom sembolünü al
            if i in [self.start_node_idx, self.end_node_idx, self.mid_node_idx]:
                circle = Circle((atom_coord[0] - 0.1, atom_coord[1]), radius=0.1, color=line_color, alpha=alpha)
                ax.add_patch(circle)
            ax.text(atom_coord[0] - 0.1, atom_coord[1], atom_symbol, fontsize=10, ha='center', va='center')
            
        
        mol_xlim = (min(atom_coord[0] for atom_coord in self.atom_coords) - 0.1, 
            max(atom_coord[0] for atom_coord in self.atom_coords) + 0.1)
        mol_ylim = (min(atom_coord[1] for atom_coord in self.atom_coords) - 0.1, 
                    max(atom_coord[1] for atom_coord in self.atom_coords) + 0.1)
        
        # Ax'in boyutunu çizimin boyutundan daha büyük ayarla
        ax.set_xlim(mol_xlim[0] - 5, mol_xlim[1] + 0.5)  # Örneğin 0.5 birim daha büyük
        ax.set_ylim(mol_ylim[0] - 0.5, mol_ylim[1] + 3.2) 
        
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+3.0, f'Method: {self.method}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+2.5, f'Smiles: {self.smiles}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+2.0, f'Prediction: {self.prediction}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+1.5, f'Sparsity: {self.sparsity:.4f}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+1.0, f'Fidelity: {self.fidelity:.4f}', fontsize=10, ha='left', va='center')
        ax.text(mol_xlim[0] - 4.8, mol_ylim[1]+0.5, f'Fidelity_inv: {self.fidelity_inv:.4f}', fontsize=10, ha='left', va='center')
        ax.set_axis_off()

        

