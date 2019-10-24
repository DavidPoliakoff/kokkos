# Kokkos Tuning

This is a design document describing the motivation, ideas, design, and prototype implementation of the Kokkos Tuning System

## Motivation

Currently, Kokkos makes a lot of decisions about tuning parameters (CUDA block sizes, different kernel implementations) 
by picking an option that results in the best performance for the widest array of applications and architectures at the 
time the choice is made. This approach leaves performance on the table, and appears increasingly untenable as the number 
of architectures and applications grows.

The Kokkos team would like to instead open up the ability to set the parameters as part of the tooling system so that
these parameters can be tuned for individual applications across all the architectures they might run on. In order to match the
feel of past Kokkos tooling efforts, we'd like to achieve this with a callback system.

## Ideas

A Kokkos Tuning system should be as small as is wise while achieving the following goals

1. Expose to tools enough data about the _context_ of the running application to tune intelligently
2. Expose to tools enough data about tuning parameters that they might know how to optimize what they're asked to
3. Expose to applications an interface that they might inform a tool about their current application context
4. Expose to tools the results of their choices
5. No perturbation of Kokkos Core when this system is disabled

Shared among the first three of these goals is a need for some way to describe the semantics of variables (tuning parameters, context variables)
internal to Kokkos or an application to an outside tool. 

### Semantics of Variables

I think it's best to talk about the semantics of variables with concrete examples.

Suppose Kokkos wants a tool to choose a block size for it. Suppose all the application context is perfectly understood, that the tool knows
that the application has 10,000,000 particles active and that it's running a kernel called "make_particles_go," which is a parallel_for in
the "cuda" execution space. The tool needs to know several things about what a block size _is_ for this to be generic and practical

1. Is it an integer value? A float? A string? (Type)
2. Relatedly, what are the mathematical semantics which are valid for it? Is it something 
for which a list can be sorted? Do the distances between items in a sorted list make sense?
If I divide two values, does the ratio have some meaning? (semantics)
3. What are the valid choices for this value? Is a block size of -128 okay? How about 7? (candidates)

Semantics (as always) are likely the source of the most confusion here, so a bit of detail is good. Here I'm leaning heavily on the field
of statistics to enable tools to do intelligent searching. If ordering doesn't make sense, if a value is "categorical", the only thing
a tool can do is try all possible values for a tuning value. If they're ordered (ordinal), the search can take advantage of this by 
using the concept of a directional search. If the distances between elements matter (interval data) you can cheat with things like
bisection. Finally if ratios matter you can play games where you increase by a factor of 10 in your searches.

In describing the candidate values in (3), users have two options: sets or ranges. A set has a number of entries of the given type, a range has lower and upper bounds and a step size.

Claim: the combination of context, candidates, semantics, and types gives a tool enough to intelligently explore the search space of 
tuning parameters

### Context

Suppose a tool perfectly understands what a block size is. To effectively tune one, it needs to know something about the application.

In a trivial case, the tool knows absolutely nothing other than candidate values, and tries to make a choice that optimizes across all
invocations of kernels. This isn't _that_ far from what Kokkos does now, so it's not unreasonable for this to produce decent results. 
That said, we could quickly add some context from Kokkos, stuff like the name and type of the kernel, the execution space, all with the semantic information described above [TODO: caveat one]. That way a tuning tool could differentiate based on all the information available to Kokkos. Going a little further, we could expose this ability to provide context to our applications. What if the tools wasn't just tuning to the fact that the kernel name was "GEMM", but that "matrix_size" was a million? Or that "live_particles" had a 
certain value? The more (relevant) context we provide to a tool, the better it will be able to tune.


### Intended Tool Workflow

Okay, so a tool knows what it's tuning, and it knows the context of the application well enough to do clever ML things, all of this with happy semantic information so that everything make . What should a workflow look like? A tool should

1) Listen to declarations about the semantics of context and tuning variables
2) Make tuning decisions
3) Measure their feedback
4) Get better at (2)

The easier we make this loop, the better

## Design

The design of this system is intended to reflect the above ideas with the minimal necessary additions to make the mechanics work. This section is almost entirely describing the small holes in the above descriptions. Variable declaration works exactly as described above, except for two things

1) Each variable is associated with a unique ID at declaration time
2) In addition to allowing "int, float, string" types, we also allow for sets and ranges of the same

(2) is important because it allows us to express interdependency of tuning variables, you can't tune "blockSize.x" and "blockSize.y" independently, the choices intersect [caveat two: interdependent variables must have the same type]. So we wouldn't describe blockSize.x as being between 32 and MAX_BLOCK_SIZE, just like blockSize.y, we would describe "3D_block_size" as being in {1,1,1}, ... {MAX_BLOCK_SIZE,1,1}

Any time a value of a variable is declared (context) or requested (tuning), it is also associated with a context ID that says how long that declaration is valid for. So if a user sees

```
declare_value("is_safe_to_push_button",true,contextId(0));
foo();
endContext(contextId(0));
bar();
```

They should know in `bar` that it is no longer safe to push the button. Similarly, if the have provided tuning values to contextId(0), when contextId(0) ends, that is when they take measurements related to those tuning values and learn things. For many tools, the first time they see a value associated with a contextId, they'll do a starting measurement, and at endContext they'll stop that measurement.

The ugliest divergence from design is in the semantics. We would absolutely love to tell users the valid values for a given tuning parameter at variable declaration time. We hate the idea of telling them the valid values on each request for the value of that parameter. Unfortunately the universe is cruel: things can happen outside of Kokkos that make the valid values of a tuning parameter change on each request. Just taking the example of block size

1) Different kernels have different valid values for block size
2) Different invocations of the same kernel can have different values for block size if somebody changes settings
3) We don't know how much worse this gets as we move past block size

So we'll do our best to mitigate the impacts of this, but for now the set of candidate values must be provided every time we request a
value

Otherwise the ideas behind the tuning system translate directly into the design and the implementation

## Implementation
