#include <stdio.h>
#include "comparetag.h"

/* comparetag benchmark.txt checkme.txt errors.txt */
main(int argc, char *argv[])
{
	if (argc != 4)
	{
		puts("Incorrect number of arguments.");
		puts("Correct format is \" comparetag benchmark.txt checkme.txt errors.txt \".");
		return 1;
	}
	
	comparetag(argv[1], argv[2], argv[3], stdout);

	return 0;
}