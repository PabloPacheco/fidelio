import numpy as np
from scipy.spatial import Delaunay
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse import coo_matrix, csr_matrix, issparse
from scipy.sparse.linalg import (
    spsolve,
    cg, bicg, bicgstab, gmres,
    lgmres, minres, qmr, gcrotmk
)
from scipy.interpolate import griddata

# ==========================================================
# Face object
# ==========================================================

@dataclass
class Face:
    id: int
    vertex: np.ndarray      # (2,)
    elements: np.ndarray    # (1 or 2,)
    center: np.ndarray      # (2,)
    area: float

# ==========================================================
# Element object
# ==========================================================

@dataclass
class Element:
    id: int
    vertex: np.ndarray
    faces: np.ndarray
    neighbors: np.ndarray
    centroid: np.ndarray
    volume: float

    # Geometría FVM (se asignan después)
    dCF: np.ndarray = None
    dCf: np.ndarray = None
    dfF: np.ndarray = None
    Sf: np.ndarray = None


# ==========================================================
# MeshFVM
# ==========================================================

class MeshFVM:

    def __init__(self, points, simplices, tess = None):
        
        if not tess == None:
            self.points = tess.points
            self.simplices = tess.simplices
            self.n_elements = self.simplices.shape[0]
        else:
            self.points = points
            self.simplices = simplices
            self.n_elements = self.simplices.shape[0]
            
        # --------------------------------------------------
        # 1) Build local faces (vectorized)
        # --------------------------------------------------
        # Para triángulos:
        # (n_elem, 3, 2)
        local_faces = self.simplices[:, [[0,1],[1,2],[2,0]]]

        # reshape → (n_elem*3, 2)
        all_faces = local_faces.reshape(-1, 2)

        # ordenar vértices (cara sin orientación)
        all_faces = np.sort(all_faces, axis=1)

        # --------------------------------------------------
        # 2) Unique global faces
        # --------------------------------------------------
        unique_faces, inv, counts = np.unique(
            all_faces,
            axis=0,
            return_inverse=True,
            return_counts=True
        )

        self.face_vertices = unique_faces  # (Nf,2) nodos ordenados
        self.n_faces = unique_faces.shape[0]

        # --------------------------------------------------
        # 3) Map face -> elements
        # --------------------------------------------------
        face_ids = inv.reshape(self.n_elements, -1)

        # elemento asociado a cada cara local
        elem_ids = np.repeat(np.arange(self.n_elements), 3)

        # --------------------------------------------
        # Map face -> elements (vectorized)
        # --------------------------------------------
        
        # elemento asociado a cada cara local
        elem_ids = np.repeat(np.arange(self.n_elements), 3)
        
        # ordenar por face id
        order = np.argsort(inv)
        inv_sorted = inv[order]
        elem_sorted = elem_ids[order]
        
        # encontrar cortes
        unique_face_ids, start_idx = np.unique(inv_sorted, return_index=True)
        
        # inicializar matriz (Nf, 2)
        face_elements = -np.ones((self.n_faces, 2), dtype=int)
        
        # número de elementos por cara (1 o 2)
        counts = np.diff(np.append(start_idx, len(inv_sorted)))
        
        # primer elemento de cada cara
        face_elements[unique_face_ids, 0] = elem_sorted[start_idx]
        
        # segundo elemento (solo donde exista)
        mask = counts == 2
        face_elements[unique_face_ids[mask], 1] = elem_sorted[start_idx[mask] + 1]


        # --------------------------------------------------
        # 4) Element neighbors
        # --------------------------------------------------
        face_ids = inv.reshape(self.n_elements, -1)
        
        # (Ne,3,2)
        elem_face_pairs = face_elements[face_ids]
        
        # elemento actual expandido
        elem_index = np.arange(self.n_elements)[:, None]
        
        # vecino = suma de ambos - elemento actual
        neighbors = elem_face_pairs.sum(axis=2) - elem_index

        # --------------------------------------------------
        # 5) Geometry
        # --------------------------------------------------
        face_points = self.points[unique_faces]
        face_center = face_points.mean(axis=1)
        face_area = np.linalg.norm(
            face_points[:,0] - face_points[:,1],
            axis=1
        )

        elem_points = self.points[self.simplices]
        centroid = elem_points.mean(axis=1)

        # área triángulo
        v1 = elem_points[:,1] - elem_points[:,0]
        v2 = elem_points[:,2] - elem_points[:,0]
        volume = 0.5 * np.abs(
            v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
        )

        # --------------------------------------------------
        # 6) Create objects
        # --------------------------------------------------
        self.faces = [
            Face(
                id=i,
                vertex=unique_faces[i],
                elements=face_elements[i],
                center=face_center[i],
                area=face_area[i]
            )
            for i in range(self.n_faces)
        ]

        self.elements = [
            Element(
                id=i,
                vertex=self.simplices[i],
                faces=face_ids[i],
                neighbors=neighbors[i],
                centroid=centroid[i],
                volume=volume[i]
            )
            for i in range(self.n_elements)
        ]


    def compute_fvm_geometry(self):
        """
        Compute geometric vectors needed for FVM diffusion.
        """
    
        Ne = self.n_elements
        Nf_loc = self.simplices.shape[1]
    
        # ---------------------------------------------
        # Centros
        # ---------------------------------------------
        elem_centroids = np.array([e.centroid for e in self.elements])
        face_centers = np.array([f.center for f in self.faces])
    
        # ---------------------------------------------
        # Expand face centers per element
        # ---------------------------------------------
        face_ids = np.array([e.faces for e in self.elements])
        neighbor_ids = np.array([e.neighbors for e in self.elements])
    
        xf = face_centers[face_ids]               # (Ne, Nf, 2)
        xc = elem_centroids[:, None, :]           # (Ne, 1, 2)
    
        # ---------------------------------------------
        # dCf
        # ---------------------------------------------
        dCf = xf - xc                             # (Ne, Nf, 2)
    
        # ---------------------------------------------
        # dCF (neighbor vector)
        # ---------------------------------------------
        dCF = np.zeros_like(dCf)
    
        internal_mask = neighbor_ids != -1
    
        dCF[internal_mask] = (
            elem_centroids[neighbor_ids[internal_mask]]
            - elem_centroids.repeat(Nf_loc, axis=0)
            .reshape(Ne, Nf_loc, 2)[internal_mask]
        )
    
        # ---------------------------------------------
        # dfF
        # ---------------------------------------------
        dfF = np.zeros_like(dCf)
    
        dfF[internal_mask] = (
            elem_centroids[neighbor_ids[internal_mask]]
            - xf[internal_mask]
        )
    
        # ---------------------------------------------
        # Sf (face normal vectors)
        # ---------------------------------------------
        face_vertices = np.array([f.vertex for f in self.faces])
        p = self.points[face_vertices]
    
        edge_vec = p[:,1] - p[:,0]
    
        Sf_candidate = np.stack(
            [edge_vec[:,1], -edge_vec[:,0]],
            axis=1
        )
    
        # expand per element
        Sf = Sf_candidate[face_ids]
    
        # Orientación usando dCF
        dot = np.sum(Sf * dCF, axis=2)
    
        sign = np.where(dot >= 0, 1.0, -1.0)
        Sf = Sf * sign[..., None]
    
        # ---------------------------------------------
        # Guardar como atributos globales
        # ---------------------------------------------
        self.dCf = dCf
        self.dCF = dCF
        self.dfF = dfF
        self.Sf  = Sf
    
        # ---------------------------------------------
        # Precalcular normas (SOLO UNA VEZ)
        # ---------------------------------------------
        self.dCf_norm = np.linalg.norm(dCf, axis = 2)
        self.dCF_norm = np.linalg.norm(dCF, axis = 2)
        self.dfF_norm = np.linalg.norm(dfF, axis = 2)
        self.Sf_norm  = np.linalg.norm(Sf,  axis = 2)
    
        # Evitar ceros exactos en distancias internas
        self.dCF_norm = np.where(
            self.dCF_norm == 0.0,
            1e-14,
            self.dCF_norm
        )
    
        # ---------------------------------------------
        # Asignar vistas a cada elemento
        # ---------------------------------------------
        for i, elem in enumerate(self.elements):
            elem.dCf       = dCf[i]
            elem.dCF       = dCF[i]
            elem.dfF       = dfF[i]
            elem.Sf        = Sf[i]
    
            elem.dCf_norm  = self.dCf_norm[i]
            elem.dCF_norm  = self.dCF_norm[i]
            elem.dfF_norm  = self.dfF_norm[i]
            elem.Sf_norm   = self.Sf_norm[i]


    def build_boundary_face_dict(self, boundary_edges):
        """
        Map Gmsh boundary edges to internal face numbering.
    
        Parameters
        ----------
        boundary_edges : dict
            {"name": array([[n0,n1], ...])}
    
        Returns
        -------
        boundary_faces : dict
            {"name": np.array([face_ids])}
        """
    
        # ---------------------------------------------
        # 1) Crear mapa (n0,n1) -> face_id
        # ---------------------------------------------
        # Las caras internas ya están ordenadas
        face_map = {
            (v0, v1): i
            for i, (v0, v1) in enumerate(self.face_vertices)
        }
    
        boundary_faces = {}
    
        # ---------------------------------------------
        # 2) Mapear edges de gmsh
        # ---------------------------------------------
        for name, edges in boundary_edges.items():
    
            # ordenar nodos para coincidir con unique_faces
            edges_sorted = np.sort(edges, axis=1)
    
            # lookup vectorizado (rápido)
            face_ids = np.fromiter(
                (face_map[(e[0], e[1])] for e in edges_sorted),
                dtype=int,
                count=edges_sorted.shape[0]
            )
    
            boundary_faces[name] = face_ids
    
        return boundary_faces


    
    def plot_topology(
        self,
        show_elements=True,
        show_faces=True,
        show_nodes=True,
        element_color="red",
        face_color="blue",
        node_color="green",
        element_size=10,
        face_size=8,
        node_size=8,
        figsize=(8, 8),
        title=None
    ):
        """
        Visualize mesh with numbering of elements, faces and nodes.
    
        Parameters
        ----------
        show_elements : bool
        show_faces : bool
        show_nodes : bool
        element_color : str
        face_color : str
        node_color : str
        element_size : int
        face_size : int
        node_size : int
        figsize : tuple
        """
    
        fig, ax = plt.subplots(figsize=figsize)
    
        # -------------------------------------------------
        # Draw mesh
        # -------------------------------------------------
        triang = mtri.Triangulation(
            self.points[:, 0],
            self.points[:, 1],
            self.simplices
        )
        ax.triplot(triang, color="black", linewidth=0.8)
    
        # -------------------------------------------------
        # Nodes numbering
        # -------------------------------------------------
        if show_nodes:
            for i, (x, y) in enumerate(self.points):
                ax.text(
                    x, y,
                    str(i),
                    color=node_color,
                    fontsize=node_size,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.6,
                        boxstyle="round,pad=0.2"
                    )
                )
    
        # -------------------------------------------------
        # Element numbering (centroids)
        # -------------------------------------------------
        if show_elements:
            for elem in self.elements:
                x, y = elem.centroid
                ax.text(
                    x, y,
                    str(elem.id),
                    color=element_color,
                    fontsize=element_size,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.6,
                        boxstyle="round,pad=0.3"
                    )
                )
    
        # -------------------------------------------------
        # Face numbering (face centers)
        # -------------------------------------------------
        if show_faces:
            for face in self.faces:
                x, y = face.center
                ax.text(
                    x, y,
                    str(face.id),
                    color=face_color,
                    fontsize=face_size,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.6,
                        boxstyle="round,pad=0.2"
                    )
                )
    
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


class Tessellation:
    """
    Planar tessellation based on Delaunay triangulation.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Node coordinates (x, y).

    Attributes
    ----------
    points : ndarray (N, 2)
        Node coordinates.
    n_nodes : int
        Number of nodes.
    tri : scipy.spatial.Delaunay
        Delaunay triangulation object.
    simplices : ndarray (Nt, 3)
        Triangle connectivity (node indices).
    connectivity : ndarray (Ne, 2)
        Edge connectivity (unique undirected edges).
    n_elements : int
        Number of elements (edges).
    lines : ndarray (Ne, 2, 2)
        Node coordinates of each edge.
    element_length : ndarray (Ne,)
        Length of each edge.
    node_id : list[int]
        Node IDs (1-based).
    element_id : list[int]
        Element IDs (1-based).
    """

    def __init__(self, points: np.ndarray, domain_filter=None):

        # -----------------------------
        # Input validation
        # -----------------------------
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be an array of shape (N, 2)")

        self.points = points
        self.n_nodes = points.shape[0]

        # -----------------------------
        # Delaunay triangulation
        # -----------------------------
        self.tri = Delaunay(points)
        self.simplices = self.tri.simplices  # (Nt, 3)

        # --- FILTRO DE DOMINIO ---
        if domain_filter is not None:
            centroids = np.mean(points[self.simplices], axis=1)
            mask = np.array([domain_filter(c) for c in centroids])
            self.simplices = self.simplices[mask]

        # -----------------------------
        # Edge extraction (unique edges)
        # -----------------------------
        edges = set()

        for tri_nodes in self.simplices:
            i, j, k = tri_nodes
            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))

        self.connectivity = np.array(list(edges), dtype=int)
        self.n_elements = self.connectiv-ity.shape[0]

        # -----------------------------
        # Edge geometry
        # -----------------------------
        self.lines = self.points[self.connectivity]  # (Ne, 2, 2)

        diff = self.lines[:, 0, :] - self.lines[:, 1, :]
        self.element_length = np.linalg.norm(diff, axis=1)

        # -----------------------------
        # Identifiers (1-based)
        # -----------------------------
        self.node_id = np.arange(0, self.n_nodes).tolist()
        self.element_id = np.arange(0, self.n_elements).tolist()



class FVMProblem:

    def __init__(self, mesh: MeshFVM):

        self.mesh = mesh
        self.Ne = mesh.n_elements
        self.Nf = mesh.n_faces

        # ----------------------------------------
        # Campos principales
        # ----------------------------------------
        self.phi_C = None          # (Ne,)
        self.phi_f = None          # (Nf,)

        self.grad_phi_C = None     # (Ne,2)
        self.grad_phi_f = None     # (Nf,2)

        # ----------------------------------------
        # Difusión
        # ----------------------------------------
        self.gamma_function = None
        self.Gamma_C = None        # (Ne,)
        self.Gamma_f = None        # (Nf,)

    
    def initialize_phi(self, phi_init):
    
        if callable(phi_init):
            self.phi_C = phi_init(self.mesh)
    
        elif np.isscalar(phi_init):
            self.phi_C = np.full(self.Ne, phi_init)
    
        else:
            self.phi_C = np.asarray(phi_init)
    
        # -------------------------------------
        # Inicializar phi_f por interpolación
        # -------------------------------------
        self._update_phi_faces()
    
        # -------------------------------------
        # Inicializar gradientes
        # -------------------------------------
        self._update_gradients()
        self._update_face_gradients()



    def _update_phi_faces(self):
    
        mesh = self.mesh
    
        # ------------------------------------------
        # Conectividad face -> (owner, neighbor)
        # ------------------------------------------
        face_elements = np.array([f.elements for f in mesh.faces])
    
        owner = face_elements[:, 0]
        neighbor = face_elements[:, 1]
    
        phi_owner = self.phi_C[owner]
    
        phi_f = np.zeros(self.Nf)
    
        internal_mask = neighbor != -1
    
        # =====================================================
        # -------- INTERIOR FACES -----------------------------
        # =====================================================
        if np.any(internal_mask):
    
            phi_neighbor = self.phi_C[neighbor[internal_mask]]
    
            # -------------------------------------------------
            # Opción 1 (activa):
            # g_f = V_C / (V_C + V_F)
            # -------------------------------------------------
            volumes = np.array([e.volume for e in mesh.elements])
    
            Vc = volumes[owner[internal_mask]]
            Vf = volumes[neighbor[internal_mask]]
    
            g_f = Vc / (Vc + Vf)
    
            # -------------------------------------------------
            # Opción 2 (geométrica - dejar comentada)
            # -------------------------------------------------
            # face_ids_per_elem = np.array([e.faces for e in mesh.elements])
            # neighbors_per_elem = np.array([e.neighbors for e in mesh.elements])
            #
            # # mapa inverso: necesitamos saber qué cara local corresponde
            # # para cada cara global. Esto requiere estructura adicional
            # # si se quiere hacer sin loops.
            #
            # Sf = mesh.Sf.reshape(-1, 2)  # requiere indexación adecuada
            # n_hat = Sf / np.linalg.norm(Sf, axis=1, keepdims=True)
            #
            # dCf = mesh.dCf.reshape(-1, 2)
            # dfF = mesh.dfF.reshape(-1, 2)
            #
            # num = np.sum(dCf * n_hat, axis=1)
            # den = num + np.sum(dfF * n_hat, axis=1)
            #
            # g_f = num / den
    
            # -------------------------------------------------
            # Interpolación
            # -------------------------------------------------
            phi_f[internal_mask] = (
                g_f * phi_neighbor +
                (1.0 - g_f) * phi_owner[internal_mask]
            )
    
        # =====================================================
        # -------- BOUNDARY FACES -----------------------------
        # =====================================================
        phi_f[~internal_mask] = phi_owner[~internal_mask]
    
        self.phi_f = phi_f



    def _update_gradients(self):
    
        Ne = self.Ne
        mesh = self.mesh
    
        face_ids = np.array([e.faces for e in mesh.elements])
        Sf = mesh.Sf                  # (Ne,Nf_loc,2)
        volumes = np.array([e.volume for e in mesh.elements])
    
        phi_f_local = self.phi_f[face_ids]   # (Ne,Nf_loc)
    
        # Expandir dimensión
        phi_f_exp = phi_f_local[..., None]   # (Ne,Nf_loc,1)
    
        grad_phi_C = np.sum(phi_f_exp * Sf, axis=1) / volumes[:, None]
    
        self.grad_phi_C = grad_phi_C
    
        # Guardar como vista en elementos
        for i, elem in enumerate(mesh.elements):
            elem.grad_phi = grad_phi_C[i]


    def _update_face_gradients(self):
        """
        Calcula grad_phi_f usando interpolación geométrica
        basada en distancias |dCf| y |dfF|.
        Requiere que grad_phi_C esté actualizado.
        """
    
        if self.grad_phi_C is None:
            raise RuntimeError("grad_phi_C must be computed first.")
    
        mesh = self.mesh
        Nf = self.Nf
    
        # ---------------------------------------------
        # Conectividad global face -> (owner, neighbor)
        # ---------------------------------------------
        face_elements = np.array([f.elements for f in mesh.faces])
    
        owner = face_elements[:, 0]
        neighbor = face_elements[:, 1]
    
        grad_owner = self.grad_phi_C[owner]
    
        grad_phi_f = np.zeros((Nf, 2))
    
        internal_mask = neighbor != -1
    
        # =====================================================
        # Caras internas
        # =====================================================
        if np.any(internal_mask):
    
            F = neighbor[internal_mask]
    
            grad_neighbor = self.grad_phi_C[F]
    
            # -------------------------------------------------
            # Obtener |dCf| y |dfF| por cara global
            # -------------------------------------------------
    
            # mesh ya tiene dCf_norm y dfF_norm por elemento-local
            # debemos construirlos en estructura global por cara
    
            # Para evitar loops grandes, usamos owner como primario
            dCf_norm_global = np.zeros(Nf)
            dfF_norm_global = np.zeros(Nf)
    
            # owner primario
            for elem_id, elem in enumerate(mesh.elements):
    
                for loc, face_id in enumerate(elem.faces):
    
                    if elem_id == owner[face_id]:
    
                        dCf_norm_global[face_id] = mesh.dCf_norm[elem_id, loc]
                        dfF_norm_global[face_id] = mesh.dfF_norm[elem_id, loc]
    
            dCf = dCf_norm_global[internal_mask]
            dfF = dfF_norm_global[internal_mask]
    
            g_f = dCf / (dCf + dfF)
    
            g_f = g_f[:, None]  # expandir dimensión
    
            grad_phi_f[internal_mask] = (
                g_f * grad_neighbor
                + (1.0 - g_f) * grad_owner[internal_mask]
            )
    
        # =====================================================
        # Caras de borde
        # =====================================================
        boundary_mask = ~internal_mask
    
        if np.any(boundary_mask):
            grad_phi_f[boundary_mask] = grad_owner[boundary_mask]
    
        self.grad_phi_f = grad_phi_f


    
    def compute_diffusion_coefficients(self):
        """
        Calcula coeficientes difusivos D_f
        únicamente para caras internas,
        usando normas geométricas precomputadas
        en MeshFVM.
        """
    
        mesh = self.mesh
    
        # -------------------------------------------------
        # Conectividad elemento → caras
        # -------------------------------------------------
        face_ids = np.array([e.faces for e in mesh.elements])       # (Ne, Nf_loc)
        neighbors = np.array([e.neighbors for e in mesh.elements])  # (Ne, Nf_loc)
    
        # -------------------------------------------------
        # Máscara de caras internas
        # -------------------------------------------------
        internal_mask = neighbors != -1
    
        # -------------------------------------------------
        # Normas ya precomputadas en mesh
        # -------------------------------------------------
        dCF_norm = mesh.dCF_norm        # (Ne, Nf_loc)
        Sf_norm  = mesh.Sf_norm         # (Ne, Nf_loc)
    
        # -------------------------------------------------
        # Gamma en caras (expandido a estructura local)
        # -------------------------------------------------
        Gamma_f_local = self.Gamma_f[face_ids]
    
        # -------------------------------------------------
        # Coeficiente difusivo local
        # -------------------------------------------------
        D_local = np.zeros_like(Sf_norm)
    
        D_local[internal_mask] = (
            Gamma_f_local[internal_mask]
            * Sf_norm[internal_mask]
            / dCF_norm[internal_mask]
        )
    
        # -------------------------------------------------
        # Construir D_f global sin duplicación
        # -------------------------------------------------
        Nf = self.Nf
        D_f = np.zeros(Nf)
    
        # Tomamos solo la contribución del primer elemento
        # asociado a cada cara (owner primario)
    
        face_elements = np.array([f.elements for f in mesh.faces])
        owner_primary = face_elements[:, 0]
    
        # Para cada elemento C, sus caras locales:
        # asignamos D_local solo si ese elemento es owner_primary
    
        for local_idx in range(face_ids.shape[1]):  # Nf_loc pequeño (3)
    
            faces_loc = face_ids[:, local_idx]
            elems = np.arange(mesh.n_elements)
                
            mask_owner = elems == owner_primary[faces_loc]
    
            mask_final = mask_owner & internal_mask[:, local_idx]
    
            D_f[faces_loc[mask_final]] = D_local[:, local_idx][mask_final]
    
        self.D_f = D_f
    

    def calculate_boundary_sources(self):
        """
        Calcula términos fuente de borde (Sc, Sp)
        asociados a elementos que poseen caras de frontera.
        """
    
        mesh = self.mesh
        Ne = self.Ne
    
        # -------------------------------------------------
        # Inicializar términos fuente
        # -------------------------------------------------
        Sc = np.zeros(Ne)
        Sp = np.zeros(Ne)
    
        # -------------------------------------------------
        # Si el usuario definió directamente Sp/Sc
        # -------------------------------------------------
        if hasattr(self, "Sc") and hasattr(self, "Sp"):
            self.Sc = self.Sc
            self.Sp = self.Sp
            return
    
        # -------------------------------------------------
        # Si existe función tipo UDF
        # -------------------------------------------------
        if hasattr(self, "boundary_sources_function") and callable(self.boundary_sources_function):
    
            Sc_user, Sp_user = self.boundary_sources_function(self)
    
            if Sc_user is not None:
                Sc += Sc_user
    
            if Sp_user is not None:
                Sp += Sp_user
    
        # -------------------------------------------------
        # Guardar resultados
        # -------------------------------------------------
        self.Sc = Sc
        self.Sp = Sp



    def initialize_gamma(self):
        """
        Inicializa Gamma_C y Gamma_f.
    
        Gamma_C puede definirse:
          - Directamente como vector (Ne,)
          - Como escalar
          - Mediante gamma_function(self)
        """
    
        mesh = self.mesh
        Ne = self.Ne
        Nf = self.Nf
    
        # =====================================================
        # 1) Construir Gamma_C
        # =====================================================
    
        if callable(self.gamma_function):
    
            Gamma_C = self.gamma_function(self)
    
        elif self.Gamma_C is not None:
    
            Gamma_C = self.Gamma_C
    
        else:
            raise ValueError("Debe definir Gamma_C o gamma_function")
    
        # ---- Permitir escalar ----
        if np.isscalar(Gamma_C):
            Gamma_C = np.full(Ne, Gamma_C)
    
        Gamma_C = np.asarray(Gamma_C)
    
        if Gamma_C.shape[0] != Ne:
            raise ValueError("Gamma_C debe tener tamaño Ne")
    
        self.Gamma_C = Gamma_C
    
    
        # =====================================================
        # 2) Construir Gamma_f
        # =====================================================
    
        # Conectividad face -> (owner, neighbor)
        face_elements = np.array([f.elements for f in mesh.faces])
    
        owner = face_elements[:, 0]
        neighbor = face_elements[:, 1]
    
        # Volúmenes por elemento
        volumes = np.array([e.volume for e in mesh.elements])
    
        Gamma_f = np.zeros(Nf)
    
        # -----------------------------------------------------
        # Caras internas
        # -----------------------------------------------------
        internal_mask = neighbor != -1
    
        if np.any(internal_mask):
    
            C = owner[internal_mask]
            F = neighbor[internal_mask]
    
            Vc = volumes[C]
            Vf = volumes[F]
    
            g_f = Vc / (Vc + Vf)
    
            Gamma_f[internal_mask] = (
                g_f * Gamma_C[F] +
                (1.0 - g_f) * Gamma_C[C]
            )
    
        # -----------------------------------------------------
        # Caras de borde
        # -----------------------------------------------------
        boundary_mask = ~internal_mask
    
        if np.any(boundary_mask):
    
            Cb = owner[boundary_mask]
    
            # Cara de borde → usar propiedad del volumen
            Gamma_f[boundary_mask] = Gamma_C[Cb]
    
        self.Gamma_f = Gamma_f


    def calculate_volumetric_sources(self):
        """
        Agrega términos fuente volumétricos linealizados.
        El usuario debe entregar Sc y Sp ya multiplicados por volumen.
        """
    
        if not hasattr(self, "Sc") or not hasattr(self, "Sp"):
            raise RuntimeError(
                "Debe ejecutar calculate_boundary_sources() primero."
            )
    
        Sc = self.Sc.copy()
        Sp = self.Sp.copy()
    
        # --------------------------------------------
        # Si existe función UDF volumétrica
        # --------------------------------------------
        if hasattr(self, "volumetric_sources_function") and callable(self.volumetric_sources_function):
    
            Sc_user, Sp_user = self.volumetric_sources_function(self)
    
            if Sc_user is not None:
                Sc += Sc_user
    
            if Sp_user is not None:
                Sp += Sp_user
    
        self.Sc = Sc
        self.Sp = Sp


    def create_cell_zone_patches(self, zones_dict):
        """
        Crea zonas volumétricas tipo:
    
        {
            "heater_zone": [cell_ids],
            "porous_zone": [cell_ids]
        }
        """
    
        mesh = self.mesh
    
        for name, cell_ids in zones_dict.items():
    
            cell_ids = np.array(cell_ids, dtype=int)
    
            patch = CellZonePatch(name)
    
            patch.cells = cell_ids
            patch.volumes = np.array([mesh.elements[i].volume for i in cell_ids])
    
            setattr(self, name, patch)

    
    def create_boundary_patches(self, bounds_dict):
        """
        Crea objetos BoundaryPatch a partir de un diccionario:
        {
            "left": [face_ids],
            "right": [face_ids]
        }
        """
    
        mesh = self.mesh
    
        for name, face_ids in bounds_dict.items():
    
            face_ids = np.array(face_ids, dtype=int)
            Nb = len(face_ids)
    
            patch = BoundaryPatch(name)
    
            cells = np.zeros(Nb, dtype=int)
            local = np.zeros(Nb, dtype=int)
    
            dCf_list = []
            dCf_norm_list = []
            Sf_list = []
            Sf_norm_list = []
    
            # ------------------------------------------
            # Recorrer caras del patch
            # ------------------------------------------
            for i, f_id in enumerate(face_ids):
    
                face = mesh.faces[f_id]
                owner = face.elements[0]
    
                elem = mesh.elements[owner]
    
                # índice local de la cara dentro del elemento
                loc = np.where(elem.faces == f_id)[0][0]
    
                cells[i] = owner
                local[i] = loc
    
                # recuperar vectores ya calculados
                dCf_list.append(mesh.dCf[owner, loc])
                dCf_norm_list.append(mesh.dCf_norm[owner, loc])
                Sf_list.append(mesh.Sf[owner, loc])
                Sf_norm_list.append(mesh.Sf_norm[owner, loc])
    
            # ------------------------------------------
            # Guardar datos en patch
            # ------------------------------------------
            patch.faces = face_ids
            patch.cells = cells
            patch.local = local
            patch.dCf = np.array(dCf_list)
            patch.dCf_norm = np.array(dCf_norm_list)
            patch.Sf = np.array(Sf_list)
            patch.Sf_norm = np.array(Sf_norm_list)
    
            # ------------------------------------------
            # Guardar como atributo dinámico
            # ------------------------------------------
            setattr(self, name, patch)

    
    def assemble_system(self):
        """
        Ensambla el sistema lineal A phi = b
        para difusión en malla no ortogonal.
        """
    
        mesh = self.mesh
        Ne = self.Ne
    
        # -------------------------------------------------
        # Geometría local
        # -------------------------------------------------
        face_ids = np.array([e.faces for e in mesh.elements])
        neighbors = np.array([e.neighbors for e in mesh.elements])
    
        Sf = mesh.Sf                    # (Ne, Nf_loc, 2)
        Sf_norm = mesh.Sf_norm
        dCF_norm = mesh.dCF_norm
        volumes = np.array([e.volume for e in mesh.elements])
    
        internal_mask = neighbors != -1
    
        # -------------------------------------------------
        # Difusión ortogonal
        # -------------------------------------------------
        Gamma_f_local = self.Gamma_f[face_ids]
    
        F_cf = np.zeros_like(Sf_norm)
    
        F_cf[internal_mask] = (
            Gamma_f_local[internal_mask]
            * Sf_norm[internal_mask]
            / dCF_norm[internal_mask]
        )
    
        # -------------------------------------------------
        # a_C
        # -------------------------------------------------
        #a_C = np.sum(F_cf, axis=1) + self.Sp * volumes
        a_C = np.sum(F_cf, axis=1) + self.Sp 
        # -------------------------------------------------
        # Ensamble off-diagonal
        # -------------------------------------------------
        rows = []
        cols = []
        data = []
    
        elem_ids = np.arange(Ne)
    
        for k in range(face_ids.shape[1]):
    
            mask = internal_mask[:, k]
    
            C = elem_ids[mask]
            F = neighbors[mask, k]
            coeff = -F_cf[mask, k]
    
            rows.append(C)
            cols.append(F)
            data.append(coeff)
    
        # -------------------------------------------------
        # Diagonal
        # -------------------------------------------------
        rows.append(elem_ids)
        cols.append(elem_ids)
        data.append(a_C)
    
        A = coo_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(Ne, Ne)
        ).tocsr()
    
        # -------------------------------------------------
        # Corrección no ortogonal
        # -------------------------------------------------
        # T_f = S_f - |S_f| e_CF
    
        dCF = mesh.dCF
        e_CF = dCF / dCF_norm[..., None]
    
        T_f = Sf - Sf_norm[..., None] * e_CF
    
        # grad_phi_f expandido local
        grad_phi_f_local = self.grad_phi_f[face_ids]
    
        Gamma_f_vec = Gamma_f_local[..., None]
    
        non_orth_term = np.sum(
            Gamma_f_vec * grad_phi_f_local * T_f,
            axis=2
        )
    
        #b_C = (
        #    self.Sc * volumes
        #    + np.sum(non_orth_term, axis=1)
        #)

        b_C = (
            self.Sc     #+ np.sum(non_orth_term, axis=1)
        )
    
        self.A = A
        self.b = b_C

    
    def solve(self, solver = "spsolve", tol = 1e-12, maxiter = 1000):
    
        if self.A is None or self.b is None:
            raise RuntimeError("Sistema no ensamblado.")
    
        A = self.A
        b = self.b
    
        # -------------------------------------------------
        # Solvers directos
        # -------------------------------------------------
        if solver == "spsolve":
            phi = spsolve(A, b)
    
        # -------------------------------------------------
        # Solvers iterativos
        # -------------------------------------------------
        elif solver == "cg":
            phi, info = cg(A, b, rtol=tol, maxiter=maxiter)
    
        elif solver == "bicg":
            phi, info = bicg(A, b, tol=tol, maxiter=maxiter)
    
        elif solver == "bicgstab":
            phi, info = bicgstab(A, b, tol=tol, maxiter=maxiter)
    
        elif solver == "gmres":
            phi, info = gmres(A, b, tol=tol, restart=50, maxiter=maxiter)
    
        elif solver == "lgmres":
            phi, info = lgmres(A, b, tol=tol, maxiter=maxiter)
    
        elif solver == "minres":
            phi, info = minres(A, b, tol=tol, maxiter=maxiter)
    
        elif solver == "qmr":
            phi, info = qmr(A, b, tol=tol, maxiter=maxiter)
    
        elif solver == "gcrotmk":
            phi, info = gcrotmk(A, b, tol=tol, maxiter=maxiter)
    
        else:
            raise ValueError(f"Solver '{solver}' no reconocido.")
    
        self.phi_C = phi
    
        # -------------------------------------------------
        # Actualizar reconstrucciones
        # -------------------------------------------------
        self._update_phi_faces()
        self._update_gradients()
        self._update_face_gradients()
    
        # -------------------------------------------------
        # Recalcular phi en fronteras usando balance
        # -------------------------------------------------
        self._update_boundary_phi_from_flux()
    
        return print("problem solved...")


    def plot_matrix_sparsity(self, figsize = (6,6), markersize = 1):
        """
        Visualiza la estructura dispersa de la matriz A
        usando matplotlib.spy.
        """
    
        if not hasattr(self, "A") or self.A is None:
            raise RuntimeError("La matriz A no está ensamblada.")
    
        if not issparse(self.A):
            raise TypeError("A debe ser una matriz dispersa scipy.")
    
        A = self.A.tocsr()
    
        plt.figure(figsize = figsize)
        plt.spy(A, markersize = markersize)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()


    def _update_boundary_phi_from_flux(self):
    
        for attr in dir(self):
    
            patch = getattr(self, attr)
    
            if not isinstance(patch, BoundaryPatch):
                continue
    
            cells = patch.cells
            faces = patch.faces
    
            Gamma_b = self.Gamma_f[faces]
            phi_C = self.phi_C[cells]
    
            grad_b = self.grad_phi_f[faces]
    
            dCb = patch.dCf_norm
            Sf = patch.Sf
            Sf_norm = patch.Sf_norm
    
            # -----------------------------------------
            # Descomposición ortogonal / tangencial
            # -----------------------------------------
            e_n = patch.dCf / dCb[:, None]
            E_b = Sf_norm
    
            T_b = Sf - E_b[:, None] * e_n
    
            # -----------------------------------------
            # Término tangencial
            # -----------------------------------------
            tangential = np.sum(grad_b * T_b, axis=1)
    
            Sp_loc = self.Sp[cells]
            Sc_loc = self.Sc[cells]
    
            numerator = (
                Sp_loc * phi_C
                + Sc_loc
                + Gamma_b * tangential
            )
    
            phi_b = phi_C - dCb / (Gamma_b * E_b) * numerator
    
            self.phi_f[faces] = phi_b
            

    def reset_sources(self):
        self.Sc = np.zeros(self.Ne)
        self.Sp = np.zeros(self.Ne)

        
    # not working :(
    def solve_nonlinear(self,
                        solver="spsolve",
                        tol=1e-8,
                        maxiter=50,
                        linear_tol=1e-8,
                        linear_maxiter=1000,
                        relaxation=1.0,
                        verbose=True):
        """
        Resuelve el problema no lineal mediante iteraciones tipo Picard.
        
        Incluye:
        - Corrección por no ortogonalidad
        - Propiedades dependientes de phi
        - Reensamble completo en cada iteración
        
        Parameters
        ----------
        solver : str
            Solver lineal scipy
        tol : float
            Tolerancia no lineal
        maxiter : int
            Máximo de iteraciones no lineales
        relaxation : float
            Factor de relajación (0<relaxation<=1)
        verbose : bool
            Imprime progreso
        """
    
        if self.phi_C is None:
            raise RuntimeError("Debe inicializar phi primero.")
    
        history = []
        phi_old = self.phi_C.copy()
    
        for it in range(1, maxiter + 1):
    
            # =====================================================
            # 1) Actualizar propiedades (Gamma puede depender de phi)
            # =====================================================
            self.initialize_gamma()
    
            # =====================================================
            # 2) Recalcular términos fuente (pueden depender de phi)
            # =====================================================
            self.reset_sources()
            self.calculate_boundary_sources()
    
            if hasattr(self, "volumetric_sources_function"):
                self.calculate_volumetric_sources()
    
            # =====================================================
            # 3) Ensamblar sistema con corrección no ortogonal
            # =====================================================
            self.assemble_system()
    
            # =====================================================
            # 4) Resolver sistema lineal
            # =====================================================
            phi_new = self.solve(
                solver=solver,
                tol=linear_tol,
                maxiter=linear_maxiter
            )
    
            # =====================================================
            # 5) Relajación
            # =====================================================
            if relaxation < 1.0:
                phi_new = (
                    relaxation * phi_new
                    + (1.0 - relaxation) * phi_old
                )
                self.phi_C = phi_new
    
            # =====================================================
            # 6) Norma de convergencia
            # =====================================================
            diff = phi_new - phi_old
    
            norm = np.linalg.norm(diff) / (
                np.linalg.norm(phi_new) + 1e-14
            )
    
            history.append(norm)
    
            if verbose:
                print(f"[NL iter {it:02d}] Residual = {norm:.3e}")
    
            if norm < tol:
                if verbose:
                    print("Convergencia no lineal alcanzada.")
                break
    
            phi_old = phi_new.copy()

            print("||Sc|| =", np.linalg.norm(self.Sc))
            print("||Sp|| =", np.linalg.norm(self.Sp))
    
        self.nonlinear_iterations = it
        self.nonlinear_residual_history = history
    
        if it == maxiter and verbose:
            print("WARNING: No convergió en el máximo de iteraciones.")
    
        return 


class BoundaryPatch:
    def __init__(self, name):
        self.name = name
        self.faces = None
        self.cells = None
        self.local = None
        self.dCf = None
        self.dCf_norm = None
        self.Sf = None
        self.Sf_norm = None

class CellZonePatch:
    def __init__(self, name):
        self.name = name
        self.cells = None      # índices globales de celdas
        self.volumes = None    # volúmenes de esas celdas


def plot_phi_contours(problem,
                      field="cell",
                      include_internal_faces=False,
                      include_boundary_faces=True,
                      levels=20,
                      resolution=200,
                      method="linear",
                      show_points=False,
                      cmap="viridis",
                      show_colorbar=True,
                      show_contour_lines=True,
                      vmin=None,
                      vmax=None,
                      alpha=1.0,
                      title = None,
                      figsize=(8, 6)):
    """
    Genera mapa de contornos interpolado de phi.

    Parameters
    ----------
    field : 'cell', 'face', 'both'
    include_internal_faces : bool
        Si True incluye caras internas.
    include_boundary_faces : bool
        Si True incluye caras de borde.
    show_contour_lines : bool
        Si False elimina líneas de contorno.
    """

    mesh = problem.mesh

    # ------------------------------------
    # Centros de celda
    # ------------------------------------
    cell_points = np.array([e.centroid for e in mesh.elements])
    cell_values = problem.phi_C

    # ------------------------------------
    # Centros de cara
    # ------------------------------------
    face_points = np.array([f.center for f in mesh.faces])
    face_values = problem.phi_f

    # detectar caras internas (neighbor != -1)
    face_elements = np.array([f.elements for f in mesh.faces])
    neighbor = face_elements[:, 1]
    internal_mask = neighbor != -1
    boundary_mask = neighbor == -1

    selected_points = []
    selected_values = []

    # ------------------------------------
    # Selección según field
    # ------------------------------------
    if field == "cell":
        selected_points.append(cell_points)
        selected_values.append(cell_values)

    elif field == "face":
        if include_internal_faces:
            selected_points.append(face_points[internal_mask])
            selected_values.append(face_values[internal_mask])

        if include_boundary_faces:
            selected_points.append(face_points[boundary_mask])
            selected_values.append(face_values[boundary_mask])

    elif field == "both":

        # Siempre incluir celdas
        selected_points.append(cell_points)
        selected_values.append(cell_values)

        if include_internal_faces:
            selected_points.append(face_points[internal_mask])
            selected_values.append(face_values[internal_mask])

        if include_boundary_faces:
            selected_points.append(face_points[boundary_mask])
            selected_values.append(face_values[boundary_mask])

    else:
        raise ValueError("field debe ser 'cell', 'face' o 'both'")

    # ------------------------------------
    # Concatenar
    # ------------------------------------
    points = np.vstack(selected_points)
    values = np.hstack(selected_values)

    x = points[:, 0]
    y = points[:, 1]

    # ------------------------------------
    # Crear malla regular
    # ------------------------------------
    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(points, values, (Xi, Yi), method=method)
    
    # Rellenar NaN en bordes usando nearest
    mask = np.isnan(Zi)
    if np.any(mask):
        Zi_nearest = griddata(points, values, (Xi, Yi), method="nearest")
        Zi[mask] = Zi_nearest[mask]

    # ------------------------------------
    # Plot
    # ------------------------------------
    plt.figure(figsize=figsize)

    contourf = plt.contourf(
        Xi, Yi, Zi,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha
    )

    if show_contour_lines:
        plt.contour(
            Xi, Yi, Zi,
            levels=levels,
            colors='k',
            linewidths=0.5
        )

    if show_points:
        plt.scatter(x, y, c='red', s=10, zorder=3)

    if show_colorbar:
        plt.colorbar(contourf, label="phi")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.gca().set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_phi_triangulation(problem,
                           levels=20,
                           show_mesh=False,
                           show_colorbar=True,
                           cmap="viridis",
                           show_contour_lines=True,
                           figsize=(8,6),
                           title=None):

    mesh = problem.mesh

    # ------------------------------------
    # Triangulación original
    # ------------------------------------
    points = mesh.points
    triangles = mesh.simplices

    triang = mtri.Triangulation(
        points[:,0],
        points[:,1],
        triangles
    )

    # ------------------------------------
    # Valor en nodos
    # ------------------------------------
    # Si no tienes phi nodal, lo interpolamos
    # desde centros de celda usando promedio simple

    phi_nodes = np.zeros(points.shape[0])
    counts = np.zeros(points.shape[0])

    for elem, phi_val in zip(mesh.elements, problem.phi_C):
        for v in elem.vertex:
            phi_nodes[v] += phi_val
            counts[v] += 1

    phi_nodes /= counts

    # ------------------------------------
    # Plot
    # ------------------------------------
    plt.figure(figsize=figsize)

    contourf = plt.tricontourf(
        triang,
        phi_nodes,
        levels=levels,
        cmap=cmap
    )

    if show_contour_lines:
        plt.tricontour(
            triang,
            phi_nodes,
            levels=levels,
            colors="k",
            linewidths=0.4
        )

    if show_mesh:
        plt.triplot(triang, color='gray', linewidth=0.5)

    if show_colorbar:
        plt.colorbar(contourf, label="phi")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.tight_layout()
    plt.show()




def read_msh_extract_data(filename, gmsh):

    gmsh.initialize()
    gmsh.open(filename)

    model = gmsh.model

    # =====================================================
    # NODOS
    # =====================================================
    node_tags, node_coords, _ = model.mesh.getNodes()

    nodes = node_coords.reshape(-1, 3)[:, :2]

    # Mapeo tag -> índice 0-based
    tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

    # =====================================================
    # TRIÁNGULOS (2D)
    # =====================================================
    elem_types, elem_tags, elem_node_tags = model.mesh.getElements(dim=2)

    triangles = []

    for etype, enodes in zip(elem_types, elem_node_tags):
        if etype == 2:  # triángulo lineal
            enodes = enodes.reshape(-1, 3)
            for tri in enodes:
                triangles.append(
                    [tag_to_index[n] for n in tri]
                )

    triangles = np.array(triangles, dtype=int)

    # =====================================================
    # BORDES: nodos y edges
    # =====================================================
    boundary_nodes = {}
    boundary_edges = {}

    phys_groups = model.getPhysicalGroups(dim=1)

    for dim, phys_tag in phys_groups:

        name = model.getPhysicalName(dim, phys_tag)
        entities = model.getEntitiesForPhysicalGroup(dim, phys_tag)

        node_set = set()
        edge_list = []

        for ent in entities:

            elem_types, elem_tags, elem_node_tags = \
                model.mesh.getElements(dim=1, tag=ent)

            for etype, enodes in zip(elem_types, elem_node_tags):

                if etype == 1:  # segmento lineal 2 nodos

                    enodes = enodes.reshape(-1, 2)

                    for edge in enodes:

                        n0 = tag_to_index[edge[0]]
                        n1 = tag_to_index[edge[1]]

                        edge_list.append([n0, n1])
                        node_set.add(n0)
                        node_set.add(n1)

        boundary_nodes[name] = np.array(sorted(node_set), dtype=int)
        boundary_edges[name] = np.array(edge_list, dtype=int)

    gmsh.finalize()

    return nodes, triangles, boundary_nodes, boundary_edges


