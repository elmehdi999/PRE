// Carré parfait de côté 1.0
L = 1.0;
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, L, 0};
Point(4) = {0, L, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Groupes physiques (le nom "Elements" est obligatoire pour GIREF)
Physical Surface("Elements") = {1};
Physical Curve("bord_bas") = {1};
Physical Curve("bord_droit") = {2};
Physical Curve("bord_haut") = {3};
Physical Curve("bord_gauche") = {4};