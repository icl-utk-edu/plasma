% Checks PLASMA_xTRADD functionality
%
% @author  Maksims Abalenkovs
% @email   m.abalenkovs@manchester.ac.uk
% @date    Sep 11, 2016
% @version 0.1

% n = 5;

% A = rand(n,n);  A = tril(A)
% B = rand(n,n)
% 
% dlmwrite('A.mtrx', A, 'delimiter', '\t', 'precision', '%.5f');
% dlmwrite('B.mtrx', B, 'delimiter', '\t', 'precision', '%.5f');

A = [1, 5, 9, 13; 2, 6, 10, 14; 3, 7, 11, 15; 4, 8, 12, 16]
B = A;

A = A'
A = tril(A)


alpha = 1.0;
beta  = alpha;

C = alpha*A + beta*B

% dlmwrite('C.mtrx',C, 'delimiter', '\t', 'precision', '%.5f');