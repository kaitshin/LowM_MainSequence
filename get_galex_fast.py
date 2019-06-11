from astropy.io import ascii as asc

FULL_PATH = '/Users/cly/Google Drive/NASA_Summer2015/'

corr_tbl = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl (1).txt',
                    guess=False, Reader=asc.FixedWidthTwoLine)

ID = corr_tbl['ID'].data

def main():

    str0 = [''] * len(ID)
    str1 = [''] * len(ID)

    for ii in range(len(ID)):
        str0[ii] = 'tar -xzvf %sFAST/outputs/BEST_FITS.tar.gz BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX_%i.fit\n' % (FULL_PATH, ID[ii])

        str1[ii] = 'tar -xzvf %sFAST/outputs/BEST_FITS.tar.gz BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX_%i.input_res.fit\n' % (FULL_PATH, ID[ii])

        str0[ii] = str0[ii].replace('Google ','Google\ ')
        str1[ii] = str1[ii].replace('Google ','Google\ ')
        
    f0 = open(FULL_PATH+'FAST/outputs/galex_fit.sh', 'w')
    f0.writelines(str0)
    f0.close()

    f1 = open(FULL_PATH+'FAST/outputs/galex_input_res.sh', 'w')
    f1.writelines(str1)
    f1.close()
    

