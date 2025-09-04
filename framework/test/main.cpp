#include "vk1/core/context.hpp"

// sdl2 doesn't allow named int main() which is redefined in it
int main(int argc, char* argv[]) {
  vk1::Context context;
  context.init();
  context.run();
  context.cleanup();
  return EXIT_SUCCESS;
}
