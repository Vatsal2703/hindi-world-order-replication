
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "andrew.h"
#include "commandline.h"
#include "unicode.h"

/* program to tag Urdu using system calls to Unitag */

/* makes it all simple like! */

/* requires to run:

	Unitag.exe
	Verticalise.exe
	Urdutag.exe
	Unirule.exe
	Deverticalise.exe
	Urdulexicon.txt
	Urdurules.txt
	Urduinst.txt

	*/

/* urduwrap listsource_filename */

#define WINSIZE 10

main(int argc, char *argv[])
{
	char infile[260];
	char outfile[260];
	char buffer[600];

	char *temp1_filename = "___uwrap_ifile1.txt";
	char *temp2_filename = "___uwrap_ifile1.txt_utg.txt";
	char *temp3_filename = "___uwrap_ifile3.txt";



	unichar uniBODY[] = { 0x003c, 0x0062, 0x006f, 0x0064, 0x0079, 0x003e, 0x0000 };


	unichar window[WINSIZE];

	FILE *listsource;
	FILE *source;
	FILE *dest;
	FILE *tempdest;

	int i;
	int end = NO;

	if (argc != 2)
	{
		fputs("Incorrect number of arguments!", stderr);
		return 1;
	}
	if( !(listsource = fopen(argv[1], "r") ) )
	{
		fprintf(stderr, "File %s couldn't be opened! Check file exists.", argv[2]);
		fcloseall();
		return 1;
	}




	while (1)
	{
		/* read a line from listsource, if none, then break */
		if (!fgets(infile, sizeof(infile), listsource))
			break;

		/* remove the line break character at the end */
		for (i = 0 ; *(infile+i) != 0 ; i++)
		{
			if (*(infile+i) == 0x0a)
				*(infile+i) = 0;
		}

		//if ( *infile ==)
		//	break


		/* change file extension for output filename */
		strcpy(outfile, infile);
		for ( i = 0 ; outfile[i] ; i++ )
			;
		outfile[i-4] = 0;
		strcat(outfile, "-tgd.txt");


		// open source
		if( !(source = fopen(infile, "rb") ) )
		{
			fprintf(stderr, "File %s couldn't be opened! Check file exists.", argv[2]);
			fcloseall();
			return 1;
		}
		
		// send header straight to end file (DEST)
		if ( !(dest = fopen(outfile, "wb")) )
		{
			cl_error_file_open(outfile);
			fcloseall();
			return 1;
		}


		for (i = 0 ; i < WINSIZE ; i++)
			window[i] = 0;

		while (1)
		{
			// write bottom character if not zero
			if (window[0])
			{
				if (fputuc(window[0], dest) == UERR)
				{
					error_file_write(outfile);
					fcloseall();
					return 1;
				}
			}

			// scroll the window down
			for (i = 0 ; i < (WINSIZE-1) ; i++)
				window[i] = window[i+1];

			// load a char to WINSIZE-1
			// if end of file occurs, it's all gone badly wrong!
			if ((window[WINSIZE-1] = fgetuc(source)) == UERR)
			{
				printf("WARNING: potential error in file %s!\n", outfile);
				break;
			}


			// break if correct point found
			if (ustrnident(window, uniBODY) == ustrlen(uniBODY))
				break;
		}

		// either a read error or <body> found
		// so direct the remainder of the source file to another file
		// open other file
		if ( !(tempdest = fopen(temp1_filename, "wb")) )
		{
			cl_error_file_open(temp1_filename);
			fcloseall();
			return 1;
		}
		// insert feff
		if (fputuc(0xfeff, tempdest) == UERR)
		{
			error_file_write(temp1_filename);
			fcloseall();
			return 1;
		}

		while (1)
		{
			// write bottom character if not zero
			if (window[0])
			{
				if (fputuc(window[0], tempdest) == UERR)
				{
					error_file_write(temp1_filename);
					fcloseall();
					return 1;
				}
			}
			else
				break;

			// scroll the window down
			for (i = 0 ; i < (WINSIZE-1) ; i++)
				window[i] = window[i+1];

			// load a char to WINSIZE-1
			// if end of file occurs, put a zero there
			if ((window[WINSIZE-1] = fgetuc(source)) == UERR)
				window[WINSIZE-1] = 0;
		}
		if (fclose(source))
		{
			cl_error_file_close(infile);
			fcloseall();
			return 1;
		}
		if (fclose(tempdest))
		{
			cl_error_file_close(temp1_filename);
			fcloseall();
			return 1;
		}

		// so now, the open dest file contains the header, and the rest of the file
		// is in temp1_filename
		// run Unitag on rest of file

		sprintf(buffer, "unitag urduinst.txt \"%s\"", temp1_filename);
		system(buffer);
		// the output will be temp2_filename


		// deverticalise it
		sprintf(buffer, "deverticalise \"%s\" \"%s\"" , temp2_filename, temp3_filename);
		system(buffer);
		// the output will be temp3_filename


		// open it as source, and send all to dest
		if( !(source = fopen(temp3_filename, "rb") ) )
		{
			fprintf(stderr, "File %s couldn't be opened! Check file exists.", argv[2]);
			fcloseall();
			return 1;
		}
		// don't bother skipping feff - it's in a place where it can do no harm 

		while (1)
		{
			if ((window[0] = fgetuc(source)) == UERR)
				break;

			if (fputuc(window[0], dest) == UERR)
				return 1;
		}
		if (fclose(source))
		{
			cl_error_file_close(temp3_filename);
			fcloseall();
			return 1;
		}
		if (fclose(dest))
		{
			cl_error_file_close(outfile);
			fcloseall();
			return 1;
		}

		/* delete all temporary files */
		remove(temp1_filename);
		remove(temp2_filename);
		remove(temp3_filename);
		
	}

	if (fclose(listsource) < 0)
	{
		fputs("Error closing file!", stderr);
		fcloseall();
		return 1;
	}

	return 0;
}