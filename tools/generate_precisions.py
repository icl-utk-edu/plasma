#! /usr/bin/env python
# -*- encoding: ascii -*-

"To be executed from the top most directory where 'tools/codegen.py' is available."

import os
import sys

def codegen(letters, filenames, fn_format):
    for filename in filenames.split():
        for letter in letters.split():
            os.system("python tools/codegen.py -p {} {}".format(letter, fn_format.format(filename)))

def main(argv):
    codegen("s d c", "plasma_z plasma_internal_z core_lapack_z plasma_core_blas_z", "include/{}.h")
    codegen("ds", "include/plasma_zc.h include/plasma_internal_zc.h include/plasma_core_blas_zc.h test/test_zc.h", "{}")
    codegen("s d c", "dzamax zgelqf zgemm zgeqrf zunglq zungqr zunmlq zunmqr zpotrf zpotrs zsymm zsyr2k zsyrk ztradd ztrmm ztrsm ztrtri zunglq zungqr zunmlq zunmqr zgbsv zgbtrf zgbtrs zgeadd zgeinv zgelqs zgels zgeqrs zgesv zgeswp zgetrf zgetri zgetrs zhemm zher2k zherk zhesv zhetrf zhetrs zlacpy zlangb zlange zlanhe zlansy zlantr zlascl zlaset zlauum zpbsv zpbtrf zpbtrs zpoinv zposv zpotri zgetri_aux zdesc2ge zdesc2pb zdesc2tr zge2desc zpb2desc ztr2desc pdzamax pzgbtrf pzgeadd pzgelqf pzgelqf_tree pzgemm pzgeqrf pzgeqrf_tree pzgeswp pzgetrf pzgetri_aux pzhemm pzher2k pzherk pzhetrf_aasen pzlacpy pzlangb pzlange pzlanhe pzlansy pzlantr pzlascl pzlaset pzlauum pzpbtrf pzpotrf pzsymm pzsyr2k pzsyrk pztbsm pztradd pztrmm pztrsm pztrtri pzunglq pzunglq_tree pzungqr pzungqr_tree pzunmlq pzunmlq_tree pzunmqr pzunmqr_tree pzdesc2ge pzdesc2pb pzdesc2tr pzge2desc pzpb2desc pztr2desc", "compute/{}.c")
    codegen("ds", "zcposv zcgesv zcgbsv clag2z zlag2c pclag2z pzlag2c", "compute/{}.c")
    codegen("s d c", "zgeadd zgemm zgeswp zgetrf zheswp zlacpy zlacpy_band zheswp ztrsm dzamax zgelqt zgeqrt zgessq zhegst zhemm zher2k zherk zhessq zlange zlanhe zlansy zlantr zlascl zlaset zlauum zunmlq zunmqr zpemv zpamm zpotrf zhegst zsymm zsyr2k zsyrk zsyssq ztradd ztrmm ztrssq ztrtri ztslqt ztsmlq ztsmqr ztsqrt zttlqt zttmlq zttmqr zttqrt zunmlq zunmqr zparfb dcabs1", "core_blas/core_{}.c")
    codegen("ds", "zlag2c clag2z", "core_blas/core_{}.c")
    codegen("s d c", "z.h dzamax.c zgbsv.c zgbtrf.c zgeadd.c zgeinv.c zgelqf.c zgelqs.c zgels.c zgemm.c zgeqrf.c zgeqrs.c zgesv.c zgeswp.c zgetrf.c zgetri_aux.c zgetri.c zgetrs.c zhemm.c zher2k.c zherk.c zhesv.c zhetrf.c zlacpy.c zlangb.c zlange.c zlanhe.c zlansy.c zlantr.c zlascl.c zlaset.c zlauum.c zpbsv.c zpbtrf.c zpoinv.c zposv.c zpotrf.c zpotri.c zpotrs.c zsymm.c zsyr2k.c zsyrk.c ztradd.c ztrmm.c ztrsm.c ztrtri.c zunmlq.c zunmqr.c", "test/test_{}")
    codegen("ds", "zcposv zcgesv zcgbsv zlag2c clag2z", "test/test_{}.c")
    return 0

if "__main__" == __name__:
    sys.exit(main(sys.argv))
