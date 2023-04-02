# https://github.com/sympy/sympy/blob/26f7bdbe3f860e7b4492e102edec2d6b429b5aaf/sympy/matrices/eigen.py

def _jordan_form(M, calc_transform=True, *, chop=False):
    """Return $(P, J)$ where $J$ is a Jordan block
    matrix and $P$ is a matrix such that $M = P J P^{-1}$
    Parameters
    ==========
    calc_transform : bool
        If ``False``, then only $J$ is returned.
    chop : bool
        All matrices are converted to exact types when computing
        eigenvalues and eigenvectors.  As a result, there may be
        approximation errors.  If ``chop==True``, these errors
        will be truncated.
    Examples
    ========
    >>> from sympy import Matrix
    >>> M = Matrix([[ 6,  5, -2, -3], [-3, -1,  3,  3], [ 2,  1, -2, -3], [-1,  1,  5,  5]])
    >>> P, J = M.jordan_form()
    >>> J
    Matrix([
    [2, 1, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 1],
    [0, 0, 0, 2]])
    See Also
    ========
    jordan_block
    """

    if not M.is_square:
        raise NonSquareMatrixError("Only square matrices have Jordan forms")

    mat        = M
    has_floats = M.has(Float)

    if has_floats:
        try:
            max_prec = max(term._prec for term in M.values() if isinstance(term, Float))
        except ValueError:
            # if no term in the matrix is explicitly a Float calling max()
            # will throw a error so setting max_prec to default value of 53
            max_prec = 53

        # setting minimum max_dps to 15 to prevent loss of precision in
        # matrix containing non evaluated expressions
        max_dps = max(prec_to_dps(max_prec), 15)

    def restore_floats(*args):
        """If ``has_floats`` is `True`, cast all ``args`` as
        matrices of floats."""

        if has_floats:
            args = [m.evalf(n=max_dps, chop=chop) for m in args]
        if len(args) == 1:
            return args[0]

        return args

    # cache calculations for some speedup
    mat_cache = {}

    def eig_mat(val, pow):
        """Cache computations of ``(M - val*I)**pow`` for quick
        retrieval"""

        if (val, pow) in mat_cache:
            return mat_cache[(val, pow)]

        if (val, pow - 1) in mat_cache:
            mat_cache[(val, pow)] = mat_cache[(val, pow - 1)].multiply(
                    mat_cache[(val, 1)], dotprodsimp=None)
        else:
            mat_cache[(val, pow)] = (mat - val*M.eye(M.rows)).pow(pow)

        return mat_cache[(val, pow)]

    # helper functions
    def nullity_chain(val, algebraic_multiplicity):
        """Calculate the sequence  [0, nullity(E), nullity(E**2), ...]
        until it is constant where ``E = M - val*I``"""

        # mat.rank() is faster than computing the null space,
        # so use the rank-nullity theorem
        cols    = M.cols
        ret     = [0]
        nullity = cols - eig_mat(val, 1).rank()
        i       = 2

        while nullity != ret[-1]:
            ret.append(nullity)

            if nullity == algebraic_multiplicity:
                break

            nullity  = cols - eig_mat(val, i).rank()
            i       += 1

            # Due to issues like #7146 and #15872, SymPy sometimes
            # gives the wrong rank. In this case, raise an error
            # instead of returning an incorrect matrix
            if nullity < ret[-1] or nullity > algebraic_multiplicity:
                raise MatrixError(
                    "SymPy had encountered an inconsistent "
                    "result while computing Jordan block: "
                    "{}".format(M))

        return ret

    def blocks_from_nullity_chain(d):
        """Return a list of the size of each Jordan block.
        If d_n is the nullity of E**n, then the number
        of Jordan blocks of size n is
            2*d_n - d_(n-1) - d_(n+1)"""

        # d[0] is always the number of columns, so skip past it
        mid = [2*d[n] - d[n - 1] - d[n + 1] for n in range(1, len(d) - 1)]
        # d is assumed to plateau with "d[ len(d) ] == d[-1]", so
        # 2*d_n - d_(n-1) - d_(n+1) == d_n - d_(n-1)
        end = [d[-1] - d[-2]] if len(d) > 1 else [d[0]]

        return mid + end

    def pick_vec(small_basis, big_basis):
        """Picks a vector from big_basis that isn't in
        the subspace spanned by small_basis"""

        if len(small_basis) == 0:
            return big_basis[0]

        for v in big_basis:
            _, pivots = M.hstack(*(small_basis + [v])).echelon_form(
                    with_pivots=True)

            if pivots[-1] == len(small_basis):
                return v

    # roots doesn't like Floats, so replace them with Rationals
    if has_floats:
        from sympy.simplify import nsimplify
        mat = mat.applyfunc(lambda x: nsimplify(x, rational=True))

    # first calculate the jordan block structure
    eigs = mat.eigenvals()

    # Make sure that we have all roots in radical form
    for x in eigs:
        if x.has(CRootOf):
            raise MatrixError(
                "Jordan normal form is not implemented if the matrix have "
                "eigenvalues in CRootOf form")

    # most matrices have distinct eigenvalues
    # and so are diagonalizable.  In this case, don't
    # do extra work!
    if len(eigs.keys()) == mat.cols:
        blocks     = list(sorted(eigs.keys(), key=default_sort_key))
        jordan_mat = mat.diag(*blocks)

        if not calc_transform:
            return restore_floats(jordan_mat)

        jordan_basis = [eig_mat(eig, 1).nullspace()[0]
                for eig in blocks]
        basis_mat    = mat.hstack(*jordan_basis)

        return restore_floats(basis_mat, jordan_mat)

    block_structure = []

    for eig in sorted(eigs.keys(), key=default_sort_key):
        algebraic_multiplicity = eigs[eig]
        chain = nullity_chain(eig, algebraic_multiplicity)
        block_sizes = blocks_from_nullity_chain(chain)

        # if block_sizes =       = [a, b, c, ...], then the number of
        # Jordan blocks of size 1 is a, of size 2 is b, etc.
        # create an array that has (eig, block_size) with one
        # entry for each block
        size_nums = [(i+1, num) for i, num in enumerate(block_sizes)]

        # we expect larger Jordan blocks to come earlier
        size_nums.reverse()

        block_structure.extend(
            [(eig, size) for size, num in size_nums for _ in range(num)])

    jordan_form_size = sum(size for eig, size in block_structure)

    if jordan_form_size != M.rows:
        raise MatrixError(
            "SymPy had encountered an inconsistent result while "
            "computing Jordan block. : {}".format(M))

    blocks     = (mat.jordan_block(size=size, eigenvalue=eig) for eig, size in block_structure)
    jordan_mat = mat.diag(*blocks)

    if not calc_transform:
        return restore_floats(jordan_mat)

    # For each generalized eigenspace, calculate a basis.
    # We start by looking for a vector in null( (A - eig*I)**n )
    # which isn't in null( (A - eig*I)**(n-1) ) where n is
    # the size of the Jordan block
    #
    # Ideally we'd just loop through block_structure and
    # compute each generalized eigenspace.  However, this
    # causes a lot of unneeded computation.  Instead, we
    # go through the eigenvalues separately, since we know
    # their generalized eigenspaces must have bases that
    # are linearly independent.
    jordan_basis = []

    for eig in sorted(eigs.keys(), key=default_sort_key):
        eig_basis = []

        for block_eig, size in block_structure:
            if block_eig != eig:
                continue

            null_big   = (eig_mat(eig, size)).nullspace()
            null_small = (eig_mat(eig, size - 1)).nullspace()

            # we want to pick something that is in the big basis
            # and not the small, but also something that is independent
            # of any other generalized eigenvectors from a different
            # generalized eigenspace sharing the same eigenvalue.
            vec      = pick_vec(null_small + eig_basis, null_big)
            new_vecs = [eig_mat(eig, i).multiply(vec, dotprodsimp=None)
                    for i in range(size)]

            eig_basis.extend(new_vecs)
            jordan_basis.extend(reversed(new_vecs))

    basis_mat = mat.hstack(*jordan_basis)

    return restore_floats(basis_mat, jordan_mat)