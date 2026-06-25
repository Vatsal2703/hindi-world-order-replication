#include <stdio.h>
#include <string.h>
#include "commandline.h"
#include "verticalise.h"


/* arguments:
	1) input filename
	2) output filename
	2) (optional) a 3-character pre-inserted responsibility tag (if none, default is used)
	*/

main (int argc, char *argv[])
{
	char user[4];
	FILE *test;
	char *input_filename;
	char *output_filename;


	strcpy(user, "*US");

	/* argument testing */
	switch (argc)
	{
	case 3:
		break;
	case 4:
		if (strlen(argv[3]) != strlen(user))
		{
			puts("The specified username (3rd argument) must consist of exactly 3 characters.");
			return 1;
		}
		else
			strcpy(user, argv[3]);
		break;
	default:
		cl_error_arguments(4, argc);
		return 1;
	}
	input_filename = argv[1];
	output_filename = argv[2];



	/* does this file  exist ? */
	if ( !(test = fopen(input_filename, "rb")) )
	{
		printf("The input file \"%s\" does not exist or cannot be opened.\n", input_filename);
		puts("It has not been verticalised.");
		return 1;
	}
	else if (fclose(test) != 0)
	{
		cl_error_file_close(input_filename);
		fcloseall();
		fputs("A test file has failed to close -- program aborting.", stderr);
		return 1;
	}


	switch (verticalise(input_filename, output_filename, user))
	{
	case SUCCESS:
		break;
	case ERROR_OPEN_INPUT:
		cl_error_file_open(input_filename);
		break;
	case ERROR_READ_INPUT:
		cl_error_file_read(input_filename);
		break;
	case ERROR_CLOSE_INPUT:
		cl_error_file_close(input_filename);
		break;
	case ERROR_OPEN_OUTPUT:
		cl_error_file_open(output_filename);
		break;
	case ERROR_WRITE_OUTPUT:
		cl_error_file_write(output_filename);
		break;
	case ERROR_CLOSE_OUTPUT:
		cl_error_file_close(output_filename);
		break;
	case ERROR_MALLOC:
		cl_error_memory_alloc();
		fputs("Program aborting due to memory error.", stderr);
		fcloseall();
		return 1;
	case NONUNICODE:
		printf("File \"%s\" may not be a Unicode text file.\nThe output may contain errors.", input_filename);
		break;
	}

	return 0;
}
