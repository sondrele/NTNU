	.file	"hello.c"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	$0, %eax
	call	hello
	movl	$0, %eax
	popq	%rbp
	ret
.LFE0:
	.size	main, .-main
	.section	.rodata
.LC0:
	.string	"Hello World!"
	.text
	.globl	hello
	.type	hello, @function
hello:
.LFB1:
	pushq	%rbp
	movq	%rsp, %rbp
	movl	$.LC0, %edi
	call	puts
	movl	$0, %eax
	popq	%rbp
	ret
.LFE1:
	.size	hello, .-hello
	.ident	"GCC: (Ubuntu/Linaro 4.7.2-2ubuntu1) 4.7.2"
	.section	.note.GNU-stack,"",@progbits
