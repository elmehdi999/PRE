MeshMshFileVersion = 2.2;
Mesh.SaveAll = 0;        // Sauvegarde UNIQUEMENT les groupes physiques
Mesh.Renumber = 1;       // Force une renumérotation stricte et unique des éléments
// Paramètres géométriques
taille_maille = 0.05;
cote = 1.0;     // Plaque de 1m x 1m
rayon = 0.15;   // Trou de 30cm de diamètre au centre

// --- 1. POINTS DU CARRÉ EXTÉRIEUR ---
Point(1) = {0, 0, 0, taille_maille};
Point(2) = {cote, 0, 0, taille_maille};
Point(3) = {cote, cote, 0, taille_maille};
Point(4) = {0, cote, 0, taille_maille};

// --- 2. LIGNES DU CARRÉ ---
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4}; // Contour extérieur

// --- 3. POINTS ET ARCS DU CERCLE CENTRAL ---
Point(5) = {0.5, 0.5, 0, taille_maille}; // Centre
Point(6) = {0.5+rayon, 0.5, 0, taille_maille};
Point(7) = {0.5, 0.5+rayon, 0, taille_maille};
Point(8) = {0.5-rayon, 0.5, 0, taille_maille};
Point(9) = {0.5, 0.5-rayon, 0, taille_maille};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};
Curve Loop(2) = {5, 6, 7, 8}; // Contour du trou

// --- 4. CRÉATION DE LA SURFACE (Carré moins le cercle) ---
Plane Surface(1) = {1, 2};

// --- 5. LABELS PHYSIQUES POUR MEF++ ---
Physical Curve("bord_exterieur") = {1, 2, 3, 4};
Physical Curve("bord_trou") = {5, 6, 7, 8};
Physical Surface("Elements") = {1};

Coherence Mesh;